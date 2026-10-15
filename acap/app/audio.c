#include "audio.h"

#include <glib.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <time.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#pragma GCC diagnostic pop

#define RING_LEN (AUDIO_RATE * 8)

struct audio {
    struct pw_thread_loop* loop;
    struct pw_context* context;
    struct pw_core* core;
    struct pw_registry* registry;
    struct spa_hook registry_listener;
    struct pw_stream* stream;
    struct spa_hook stream_listener;
    uint32_t target_id;
    struct spa_audio_info_raw fmt;

    /* resampler state (only used if PipeWire does not deliver 16 kHz) */
    double rs_phase;
    float rs_prev;
    double rs_acc;
    uint32_t rs_acc_n;

    GMutex lock;
    GCond cond;
    float* inject; /* test audio replacing the mic while inject_pos < inject_len */
    float* inject_done; /* finished buffer, freed outside the realtime callback */
    size_t inject_len;
    size_t inject_pos;
    float ring[RING_LEN];
    uint64_t total;
    audio_info_t info;
};

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Called with a->lock held. */
static void ring_push(audio_t* a, float v) {
    if (a->inject) {
        v = a->inject[a->inject_pos++];
        if (a->inject_pos >= a->inject_len) {
            a->inject_done = a->inject;
            a->inject      = NULL;
        }
    }
    a->ring[a->total % RING_LEN] = v;
    a->total++;
}

/* Called with a->lock held. Converts one native-rate mono sample to 16 kHz. */
static void resample_push(audio_t* a, float v) {
    uint32_t rate = a->fmt.rate ? a->fmt.rate : AUDIO_RATE;
    if (rate == AUDIO_RATE) {
        ring_push(a, v);
        return;
    }
    if (rate % AUDIO_RATE == 0) { /* integer decimation: boxcar average */
        a->rs_acc += v;
        if (++a->rs_acc_n == rate / AUDIO_RATE) {
            ring_push(a, (float)(a->rs_acc / a->rs_acc_n));
            a->rs_acc   = 0.0;
            a->rs_acc_n = 0;
        }
        return;
    }
    /* generic: linear interpolation */
    double step = (double)rate / AUDIO_RATE;
    while (a->rs_phase <= 1.0) {
        ring_push(a, (float)(a->rs_prev + (v - a->rs_prev) * a->rs_phase));
        a->rs_phase += step;
    }
    a->rs_phase -= 1.0;
    a->rs_prev = v;
}

static void on_process(void* data) {
    audio_t* a          = data;
    struct pw_buffer* b = pw_stream_dequeue_buffer(a->stream);
    if (!b)
        return;
    struct spa_buffer* buf   = b->buffer;
    const struct spa_data* d = buf->n_datas >= 1 ? &buf->datas[0] : NULL;
    if (d && d->data && d->chunk) {
        uint32_t off  = SPA_MIN(d->chunk->offset, d->maxsize);
        uint32_t size = SPA_MIN(d->chunk->size, d->maxsize - off);
        g_mutex_lock(&a->lock);
        uint32_t ch    = a->fmt.channels ? a->fmt.channels : 1;
        uint32_t n     = size / sizeof(float) / ch;
        const float* s = (const float*)((const uint8_t*)d->data + off);
        for (uint32_t i = 0; i < n; i++) {
            float v = 0.0f;
            for (uint32_t c = 0; c < ch; c++)
                v += s[i * ch + c];
            resample_push(a, v / (float)ch);
        }
        a->info.samples_total  = a->total;
        a->info.last_data_mono = mono_s();
        a->info.streaming      = true;
        g_cond_broadcast(&a->cond);
        g_mutex_unlock(&a->lock);
    }
    pw_stream_queue_buffer(a->stream, b);
}

static void on_param_changed(void* data, uint32_t id, const struct spa_pod* param) {
    audio_t* a = data;
    struct spa_audio_info info;
    if (!param || id != SPA_PARAM_Format)
        return;
    if (spa_format_parse(param, &info.media_type, &info.media_subtype) < 0 ||
        info.media_type != SPA_MEDIA_TYPE_audio || info.media_subtype != SPA_MEDIA_SUBTYPE_raw)
        return;
    spa_format_audio_raw_parse(param, &info.info.raw);
    g_mutex_lock(&a->lock);
    a->fmt                  = info.info.raw;
    a->info.native_rate     = info.info.raw.rate;
    a->info.native_channels = info.info.raw.channels;
    g_mutex_unlock(&a->lock);
    syslog(LOG_INFO, "[traindet] audio format: %u Hz, %u ch, format %u%s", info.info.raw.rate,
           info.info.raw.channels, info.info.raw.format,
           info.info.raw.rate == AUDIO_RATE ? "" : " (resampling to 16 kHz in app)");
}

static void on_state_changed(void* data, enum pw_stream_state old, enum pw_stream_state state, const char* error) {
    audio_t* a = data;
    (void)old;
    syslog(state == PW_STREAM_STATE_ERROR ? LOG_WARNING : LOG_INFO, "[traindet] audio stream %s%s%s",
           pw_stream_state_as_string(state), error ? ": " : "", error ? error : "");
    if (state != PW_STREAM_STATE_STREAMING) {
        g_mutex_lock(&a->lock);
        a->info.streaming = false;
        g_mutex_unlock(&a->lock);
    }
    /* Streams that never recover are handled by the audio watchdog in the inference
     * thread (no audio for AUDIO_WATCHDOG_S -> exit; runMode respawn restarts us). */
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = on_state_changed,
    .param_changed = on_param_changed,
    .process       = on_process,
};

static void connect_stream(audio_t* a, uint32_t id, const char* name) {
    uint8_t pod_buf[1024];
    struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(pod_buf, sizeof(pod_buf));
    const struct spa_pod* params[1];

    struct pw_properties* props =
        pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Capture", PW_KEY_MEDIA_ROLE,
                          "Production", PW_KEY_TARGET_OBJECT, name, PW_KEY_NODE_NAME, "train_detect", NULL);
    a->stream = pw_stream_new(a->core, "Train detection capture", props);
    if (!a->stream) {
        syslog(LOG_ERR, "[traindet] pw_stream_new failed");
        return;
    }
    pw_stream_add_listener(a->stream, &a->stream_listener, &stream_events, a);
    params[0] = spa_format_audio_raw_build(&builder, SPA_PARAM_EnumFormat,
                                           &SPA_AUDIO_INFO_RAW_INIT(.format   = SPA_AUDIO_FORMAT_F32,
                                                                    .rate     = AUDIO_RATE,
                                                                    .channels = 1));
    int res = pw_stream_connect(a->stream, PW_DIRECTION_INPUT, PW_ID_ANY,
                                PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS |
                                    PW_STREAM_FLAG_RT_PROCESS,
                                params, 1);
    if (res < 0) {
        syslog(LOG_ERR, "[traindet] pw_stream_connect to %s failed: %s", name, strerror(-res));
        pw_stream_destroy(a->stream);
        a->stream = NULL;
        return;
    }
    a->target_id = id;
    g_mutex_lock(&a->lock);
    snprintf(a->info.node, sizeof(a->info.node), "%s", name);
    g_mutex_unlock(&a->lock);
    syslog(LOG_INFO, "[traindet] capturing from node %s (id %u)", name, id);
}

static void registry_global(void* data, uint32_t id, uint32_t permissions, const char* type, uint32_t version,
                            const struct spa_dict* props) {
    audio_t* a = data;
    (void)permissions;
    (void)version;
    if (!spa_streq(type, PW_TYPE_INTERFACE_Node) || !props)
        return;
    const char* cls  = spa_dict_lookup(props, PW_KEY_MEDIA_CLASS);
    const char* name = spa_dict_lookup(props, PW_KEY_NODE_NAME);
    syslog(LOG_INFO, "[traindet] pipewire node id=%u class=%s name=%s", id, cls ? cls : "-", name ? name : "-");
    /* First Audio/Source wins, except that a "...Input0" node (the main mic on AXIS OS,
     * e.g. AudioDevice0Input0) replaces an earlier non-matching pick. */
    if (cls && name && strcmp(cls, "Audio/Source") == 0) {
        bool preferred = g_str_has_suffix(name, "Input0");
        if (a->stream && preferred && !g_str_has_suffix(a->info.node, "Input0")) {
            syslog(LOG_INFO, "[traindet] switching capture to preferred node %s", name);
            spa_hook_remove(&a->stream_listener);
            pw_stream_destroy(a->stream);
            a->stream = NULL;
        }
        if (!a->stream)
            connect_stream(a, id, name);
        else
            syslog(LOG_INFO, "[traindet] ignoring extra source node %s", name);
    }
}

static void registry_global_remove(void* data, uint32_t id) {
    audio_t* a = data;
    if (a->stream && id == a->target_id) {
        syslog(LOG_WARNING, "[traindet] capture node %u removed", id);
        spa_hook_remove(&a->stream_listener);
        pw_stream_destroy(a->stream);
        a->stream = NULL;
        g_mutex_lock(&a->lock);
        a->info.streaming = false;
        g_mutex_unlock(&a->lock);
    }
}

static const struct pw_registry_events registry_events = {
    PW_VERSION_REGISTRY_EVENTS,
    .global        = registry_global,
    .global_remove = registry_global_remove,
};

audio_t* audio_start(int* argc, char*** argv) {
    pw_init(argc, argv);
    audio_t* a = calloc(1, sizeof(*a));
    if (!a)
        return NULL;
    g_mutex_init(&a->lock);
    g_cond_init(&a->cond);
    a->loop = pw_thread_loop_new("traindet-audio", NULL);
    if (!a->loop)
        goto fail;
    a->context = pw_context_new(pw_thread_loop_get_loop(a->loop), NULL, 0);
    if (!a->context)
        goto fail;
    if (pw_thread_loop_start(a->loop) < 0)
        goto fail;
    pw_thread_loop_lock(a->loop);
    a->core = pw_context_connect(a->context, NULL, 0);
    if (!a->core) {
        pw_thread_loop_unlock(a->loop);
        syslog(LOG_ERR, "[traindet] cannot connect to pipewire");
        goto fail;
    }
    a->registry = pw_core_get_registry(a->core, PW_VERSION_REGISTRY, 0);
    pw_registry_add_listener(a->registry, &a->registry_listener, &registry_events, a);
    pw_thread_loop_unlock(a->loop);
    return a;
fail:
    audio_stop(a);
    return NULL;
}

void audio_stop(audio_t* a) {
    if (!a)
        return;
    if (a->loop)
        pw_thread_loop_stop(a->loop);
    if (a->stream) {
        spa_hook_remove(&a->stream_listener);
        pw_stream_destroy(a->stream);
    }
    if (a->registry) {
        spa_hook_remove(&a->registry_listener);
        pw_proxy_destroy((struct pw_proxy*)a->registry);
    }
    if (a->core)
        pw_core_disconnect(a->core);
    if (a->context)
        pw_context_destroy(a->context);
    if (a->loop)
        pw_thread_loop_destroy(a->loop);
    g_free(a->inject);
    g_free(a->inject_done);
    g_mutex_clear(&a->lock);
    g_cond_clear(&a->cond);
    free(a);
    pw_deinit();
}

bool audio_wait_latest(audio_t* a, uint64_t* cursor, size_t min_new, float* dst, size_t n, int timeout_ms) {
    gint64 deadline = g_get_monotonic_time() + (gint64)timeout_ms * 1000;
    g_mutex_lock(&a->lock);
    while (a->total < *cursor + min_new || a->total < n) {
        if (!g_cond_wait_until(&a->cond, &a->lock, deadline)) {
            g_mutex_unlock(&a->lock);
            return false;
        }
    }
    uint64_t end = a->total;
    for (size_t i = 0; i < n; i++)
        dst[i] = a->ring[(end - n + i) % RING_LEN];
    *cursor = end;
    g_mutex_unlock(&a->lock);
    return true;
}

void audio_get_info(audio_t* a, audio_info_t* info) {
    g_mutex_lock(&a->lock);
    g_free(a->inject_done);
    a->inject_done = NULL;
    *info                 = a->info;
    info->inject_remaining = a->inject ? (double)(a->inject_len - a->inject_pos) / AUDIO_RATE : 0.0;
    g_mutex_unlock(&a->lock);
}

void audio_inject(audio_t* a, const float* samples, size_t n) {
    float* copy = g_memdup2(samples, n * sizeof(float));
    g_mutex_lock(&a->lock);
    g_free(a->inject);
    g_free(a->inject_done);
    a->inject_done = NULL;
    a->inject      = copy;
    a->inject_len = n;
    a->inject_pos = 0;
    g_mutex_unlock(&a->lock);
}
