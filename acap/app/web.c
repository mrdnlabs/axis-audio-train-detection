/* FastCGI endpoints served through the device's Apache (auth + TLS inherited):
 *   status.cgi   (viewer)  GET  live state, scores, history, effective settings
 *   analyze.cgi  (admin)   POST 16 kHz mono PCM16 WAV (<= 60 s) -> per-window scores
 *                          POST ...?mode=inject -> play the WAV into the live detector
 *                          in place of the microphone (end-to-end test of events)
 */
#include <fcgiapp.h>
#include <math.h>
#include <glib.h>
#include <jansson.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <syslog.h>
#include <time.h>

#include "app.h"
#include "wav.h"

#define WEB_THREADS 2
#define MAX_UPLOAD_BYTES (60 * AUDIO_RATE * 2 + 4096)

static int listen_sock = -1;

static double r3(double v) {
    return (double)((long long)(v * 1000.0 + (v >= 0 ? 0.5 : -0.5))) / 1000.0;
}

static void reply_json(FCGX_Request* req, int code, json_t* body) {
    char* text = json_dumps(body, JSON_COMPACT);
    FCGX_FPrintF(req->out,
                 "Status: %d\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
                 "X-Content-Type-Options: nosniff\r\n\r\n%s",
                 code, text ? text : "{}");
    free(text);
}

static void reply_error(FCGX_Request* req, int code, const char* msg) {
    json_t* o = json_pack("{s:s}", "error", msg);
    reply_json(req, code, o);
    json_decref(o);
}

static json_t* event_json(const det_event_t* e) {
    json_t* label = e->label > 0 ? json_string("train") : (e->label < 0 ? json_string("false_alarm") : json_null());
    return json_pack("{s:f,s:f,s:f,s:f,s:f,s:f,s:b,s:o}", "start", r3(e->start_wall), "end", r3(e->end_wall),
                     "duration_s", r3(e->duration_s), "peak_score", r3(e->peak_score), "mean_score",
                     r3(e->mean_score), "peak_level_db", r3(e->peak_level_db), "horn", e->horn, "label", label);
}

static const char* source_name(score_source_t s) {
    return s == SCORE_YAMNET ? "yamnet" : (s == SCORE_MAX ? "max" : "head");
}

static void handle_status(FCGX_Request* req) {
    audio_info_t ai;
    audio_get_info(g_app.audio, &ai);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now_mono = (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;

    g_mutex_lock(&g_app.lock);
    const det_config_t* c = &g_app.cfg;
    json_t* top           = json_array();
    for (int i = 0; i < TOP_K; i++) {
        if (g_app.top[i].idx < 0)
            continue;
        json_array_append_new(top, json_pack("{s:i,s:s,s:f}", "index", g_app.top[i].idx, "name",
                                             g_app.class_names[g_app.top[i].idx], "score",
                                             r3(g_app.top[i].score)));
    }
    json_t* hist = json_array();
    for (int i = 0; i < g_app.det.hist_count; i++)
        json_array_append_new(hist, event_json(detector_history(&g_app.det, i)));
    json_t* current = json_null();
    if (g_app.det.state == DET_ACTIVE) {
        current = event_json(&g_app.det.cur);
        json_object_set_new(current, "elapsed_s", json_real(r3(g_app.last_infer_wall - g_app.det.cur.start_wall)));
    }
    json_t* o = json_pack(
        "{s:s,s:s,s:f,s:o,s:{s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:b,s:b,s:o},s:o,s:o,"
        "s:{s:I,s:I,s:I,s:f,s:f,s:f,s:s},"
        "s:{s:s,s:b,s:i,s:i,s:I,s:f,s:f},"
        "s:{s:b,s:i,s:f,s:f,s:f,s:f,s:i,s:b,s:f,s:s,s:f,s:f}}",
        "app", APP_NAME, "state", detector_state_name(g_app.det.state), "uptime_s", r3(now_mono - g_app.start_mono),
        "current", current,
        "scores", "head", r3(g_app.last_head), "yamnet_rail", r3(g_app.last_rail), "horn", r3(g_app.last_horn),
        "score", r3(g_app.last.score_smooth), "threshold", r3(g_app.last.threshold), "horn_threshold",
        r3(g_app.last.horn_threshold), "level_db", r3(g_app.last_level_db), "gated", g_app.last.gated, "horn_now",
        g_app.last.horn_now, "top", top,
        "history", hist, "last_horn", g_app.det.last_horn_wall > 0 ? json_real(r3(g_app.det.last_horn_wall)) : json_null(),
        "inference", "count", (json_int_t)g_app.inferences, "errors", (json_int_t)g_app.inference_errors,
        "skipped_windows", (json_int_t)g_app.skipped_windows, "latency_ms_avg", r3(g_app.latency_ms_avg),
        "latency_ms_max", r3(g_app.latency_ms_max), "last_wall", r3(g_app.last_infer_wall), "device",
        model_device(g_app.model),
        "audio", "node", ai.node, "streaming", ai.streaming, "native_rate", (int)ai.native_rate, "native_channels",
        (int)ai.native_channels, "samples", (json_int_t)ai.samples_total, "last_data_age_s",
        ai.last_data_mono > 0 ? r3(now_mono - ai.last_data_mono) : -1.0, "inject_remaining_s",
        r3(ai.inject_remaining),
        "config", "enabled", c->enabled, "sensitivity", c->sensitivity, "threshold",
        r3(detector_threshold(c->sensitivity)), "trigger_s", r3(c->trigger_s), "release_s", r3(c->release_s),
        "smoothing_s", r3(c->smoothing_s), "horn_sensitivity", c->horn_sensitivity, "horn_triggers",
        c->horn_triggers, "min_level_db", r3(c->min_level_db), "score_source", source_name(c->score_source),
        "cooldown_s", r3(c->cooldown_s), "horn_threshold", r3(detector_horn_threshold(c->horn_sensitivity)));
    json_object_set_new(o, "events_total", json_integer((json_int_t)g_app.det.events_total));
    json_object_set_new(o, "horns_total", json_integer((json_int_t)g_app.det.horns_total));
    if (g_app.last_error[0])
        json_object_set_new(o, "last_error", json_string(g_app.last_error));
    g_mutex_unlock(&g_app.lock);

    if (!o) {
        reply_error(req, 500, "failed to build status");
        return;
    }
    reply_json(req, 200, o);
    json_decref(o);
}

static double fin(double v) {
    return isfinite(v) ? r3(v) : -1.0;
}

/* CSRF guard: require a non-"simple" content type (forces a CORS preflight, which the
 * device does not grant) and, when the browser sends one, a same-host Origin. */
static bool same_origin_upload(FCGX_Request* req) {
    const char* ct = FCGX_GetParam("CONTENT_TYPE", req->envp);
    if (!ct || g_ascii_strncasecmp(ct, "audio/wav", 9) != 0)
        return false;
    const char* origin = FCGX_GetParam("HTTP_ORIGIN", req->envp);
    const char* host   = FCGX_GetParam("HTTP_HOST", req->envp);
    if (origin && host) {
        const char* p = strstr(origin, "://");
        if (!p || strcmp(p + 3, host) != 0)
            return false;
    }
    return true;
}

static void handle_analyze_locked(FCGX_Request* req);

static void handle_analyze(FCGX_Request* req) {
    static GMutex analyze_lock;
    const char* method = FCGX_GetParam("REQUEST_METHOD", req->envp);
    if (!method || strcmp(method, "POST") != 0) {
        reply_error(req, 405, "POST a 16 kHz mono PCM16 WAV body");
        return;
    }
    if (!same_origin_upload(req)) {
        reply_error(req, 403, "Content-Type must be audio/wav and the request same-origin");
        return;
    }
    if (!g_mutex_trylock(&analyze_lock)) {
        reply_error(req, 503, "another analysis is running, try again shortly");
        return;
    }
    handle_analyze_locked(req);
    g_mutex_unlock(&analyze_lock);
}

static void handle_analyze_locked(FCGX_Request* req) {
    const char* cl = FCGX_GetParam("CONTENT_LENGTH", req->envp);
    long len       = cl ? strtol(cl, NULL, 10) : 0;
    if (len <= 44 || len > MAX_UPLOAD_BYTES) {
        reply_error(req, 413, "body must be a WAV between 44 bytes and 60 s of 16 kHz mono PCM16");
        return;
    }
    unsigned char* body = malloc((size_t)len);
    if (!body) {
        reply_error(req, 500, "out of memory");
        return;
    }
    long got = 0;
    while (got < len) {
        int n = FCGX_GetStr((char*)body + got, (int)(len - got), req->in);
        if (n <= 0)
            break;
        got += n;
    }
    if (got != len) {
        free(body);
        reply_error(req, 400, "short body");
        return;
    }
    size_t n_samples = 0;
    const char* werr = NULL;
    float* samples   = wav_decode_pcm16_mono16k(body, (size_t)len, &n_samples, &werr);
    free(body);
    if (!samples) {
        reply_error(req, 415, werr);
        return;
    }
    const char* qs = FCGX_GetParam("QUERY_STRING", req->envp);
    if (qs && strcmp(qs, "mode=inject") == 0) {
        audio_inject(g_app.audio, samples, n_samples);
        syslog(LOG_INFO, "[traindet] injecting %.1f s of test audio into the live pipeline",
               (double)n_samples / AUDIO_RATE);
        json_t* o = json_pack("{s:s,s:f}", "mode", "inject", "duration_s", r3((double)n_samples / AUDIO_RATE));
        free(samples);
        reply_json(req, 200, o);
        json_decref(o);
        return;
    }
    /* Same windowing as the live path; pad short clips to one window. */
    size_t total = n_samples < WINDOW_SAMPLES ? WINDOW_SAMPLES : n_samples;
    float* buf   = calloc(total, sizeof(float));
    if (!buf) {
        free(samples);
        reply_error(req, 500, "out of memory");
        return;
    }
    memcpy(buf, samples, n_samples * sizeof(float));
    free(samples);

    json_t* wins = json_array();
    char err[160];
    static model_output_t out; /* serialized by analyze_lock in handle_analyze() */
    float max_prob = 0, sum_prob = 0;
    size_t count   = 0;
    bool failed    = false;
    for (size_t off = 0; off + WINDOW_SAMPLES <= total; off += HOP_SAMPLES) {
        if (!model_run(g_app.model, buf + off, &out, err, sizeof(err))) {
            failed = true;
            break;
        }
        class_score_t top[3];
        app_top_classes(out.scores, top, 3);
        json_t* t = json_array();
        for (int i = 0; i < 3; i++)
            if (top[i].idx >= 0)
                json_array_append_new(t, json_pack("{s:s,s:f}", "name", g_app.class_names[top[i].idx], "score",
                                                   fin(top[i].score)));
        json_array_append_new(
            wins, json_pack("{s:f,s:f,s:f,s:f,s:f,s:f,s:o}", "t", r3((double)off / AUDIO_RATE), "prob",
                            fin(out.train_prob), "rail", fin(app_max_of(out.scores, RAIL_IDX, RAIL_IDX_N)),
                            "horn", fin(app_max_of(out.scores, HORN_IDX, HORN_IDX_N)), "level_db",
                            r3(app_level_dbfs(buf + off, WINDOW_SAMPLES)), "latency_ms", r3(out.latency_ms), "top", t));
        if (out.train_prob > max_prob)
            max_prob = out.train_prob;
        sum_prob += out.train_prob;
        count++;
    }
    free(buf);
    if (failed) {
        json_decref(wins);
        reply_error(req, 500, err);
        return;
    }
    json_t* o = json_pack("{s:f,s:I,s:f,s:f,s:o}", "duration_s", r3((double)n_samples / AUDIO_RATE), "windows_n",
                          (json_int_t)count, "max_prob", fin(max_prob), "mean_prob",
                          count ? fin(sum_prob / count) : 0.0, "windows", wins);
    reply_json(req, 200, o);
    json_decref(o);
}

/* label.cgi (operator): POST ?start=<event start, unix s>&label=train|false_alarm|none
 * Requires the X-Requested-With header (not settable cross-origin without a preflight). */
static void handle_label(FCGX_Request* req) {
    const char* method = FCGX_GetParam("REQUEST_METHOD", req->envp);
    const char* xrw    = FCGX_GetParam("HTTP_X_REQUESTED_WITH", req->envp);
    if (!method || strcmp(method, "POST") != 0) {
        reply_error(req, 405, "use POST");
        return;
    }
    if (!xrw || strcmp(xrw, "traindet") != 0) {
        reply_error(req, 403, "missing X-Requested-With: traindet");
        return;
    }
    const char* qs = FCGX_GetParam("QUERY_STRING", req->envp);
    gchar** parts  = g_strsplit(qs ? qs : "", "&", 4);
    double start   = -1.0;
    int label      = 2; /* invalid */
    for (int i = 0; parts[i]; i++) {
        if (g_str_has_prefix(parts[i], "start=")) {
            char* end = NULL;
            start     = g_ascii_strtod(parts[i] + 6, &end);
            if (!end || *end != '\0' || !isfinite(start) || start <= 0)
                start = -1.0;
        } else if (strcmp(parts[i], "label=train") == 0) {
            label = 1;
        } else if (strcmp(parts[i], "label=false_alarm") == 0) {
            label = -1;
        } else if (strcmp(parts[i], "label=none") == 0) {
            label = 0;
        }
    }
    g_strfreev(parts);
    if (start < 0 || label == 2) {
        reply_error(req, 400, "need start=<unix seconds> and label=train|false_alarm|none");
        return;
    }
    g_mutex_lock(&g_app.lock);
    bool found = detector_set_label(&g_app.det, start, (int8_t)label);
    g_mutex_unlock(&g_app.lock);
    if (!found) {
        reply_error(req, 404, "no such event in history");
        return;
    }
    syslog(LOG_INFO, "[traindet] event %.3f labelled %s", start, label > 0 ? "train" : (label < 0 ? "false_alarm" : "none"));
    json_t* o = json_pack("{s:f,s:i}", "start", r3(start), "label", label);
    reply_json(req, 200, o);
    json_decref(o);
}

static gpointer web_worker(gpointer data) {
    (void)data;
    FCGX_Request req;
    if (FCGX_InitRequest(&req, listen_sock, 0) != 0) {
        syslog(LOG_ERR, "[traindet] FCGX_InitRequest failed");
        return NULL;
    }
    for (;;) {
        if (FCGX_Accept_r(&req) != 0) {
            syslog(LOG_WARNING, "[traindet] FCGX_Accept_r failed, retrying");
            g_usleep(500000);
            continue;
        }
        const char* uri = FCGX_GetParam("SCRIPT_NAME", req.envp);
        if (!uri)
            uri = FCGX_GetParam("REQUEST_URI", req.envp);
        if (uri && strstr(uri, "/status.cgi"))
            handle_status(&req);
        else if (uri && strstr(uri, "/analyze.cgi"))
            handle_analyze(&req);
        else if (uri && strstr(uri, "/label.cgi"))
            handle_label(&req);
        else
            reply_error(&req, 404, "not found");
        FCGX_Finish_r(&req);
    }
    return NULL; /* not reached */
}

bool web_start(void) {
    const char* path = getenv("FCGI_SOCKET_NAME");
    if (!path) {
        syslog(LOG_ERR, "[traindet] FCGI_SOCKET_NAME not set");
        return false;
    }
    if (FCGX_Init() != 0)
        return false;
    listen_sock = FCGX_OpenSocket(path, 8);
    if (listen_sock < 0)
        return false;
    /* Apache must be able to connect to the socket (as in the SDK fastcgi example). */
    chmod(path, S_IRWXU | S_IRWXG | S_IRWXO);
    for (int i = 0; i < WEB_THREADS; i++)
        g_thread_unref(g_thread_new("traindet-web", web_worker, NULL));
    syslog(LOG_INFO, "[traindet] web API on %s", path);
    return true;
}
