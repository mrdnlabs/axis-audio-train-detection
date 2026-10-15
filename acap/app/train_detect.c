/**
 * train_detect - on-device audio train detection for Axis devices.
 *
 * Threads:
 *   main      GLib main loop: AXParameter callbacks and AXEvent declaration/sending.
 *   audio     PipeWire thread loop: microphone -> 16 kHz ring buffer (audio.c).
 *   inference every 0.48 s: last 0.975 s -> larod (YAMNet + train head) -> detector.
 *   web       FastCGI workers: status.cgi / analyze.cgi (web.c).
 */
#include <axsdk/axevent.h>
#include <axsdk/axparameter.h>
#include <glib-unix.h>
#include <glib.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <time.h>

#include "app.h"

#define MODEL_PATH "/usr/local/packages/" APP_NAME "/model/traindet.tflite"
#define CLASSMAP_PATH "/usr/local/packages/" APP_NAME "/model/yamnet_class_map.csv"
#define LAROD_DEVICES "cpu-tflite"
#define AUDIO_WATCHDOG_S 60

app_t g_app;

const int RAIL_IDX[]  = {322, 323, 324, 325, 326, 327, 328};
const int RAIL_IDX_N  = 7;
const int HORN_IDX[]  = {324, 325};
const int HORN_IDX_N  = 2;

static GMainLoop* main_loop;
static AXEventHandler* event_handler;
static guint event_decl;
static gboolean event_declared;
static volatile gint stop_flag;

/* ---------- helpers ---------- */

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double wall_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

float app_level_dbfs(const float* x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++)
        acc += (double)x[i] * x[i];
    return (float)(10.0 * log10(acc / n + 1e-12));
}

float app_max_of(const float* scores, const int* idx, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++)
        if (scores[idx[i]] > m)
            m = scores[idx[i]];
    return m;
}

void app_top_classes(const float* scores, class_score_t* top, int k) {
    for (int j = 0; j < k; j++) {
        top[j].idx   = -1;
        top[j].score = -1.0f;
    }
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        for (int j = 0; j < k; j++) {
            if (scores[i] > top[j].score) {
                memmove(&top[j + 1], &top[j], (size_t)(k - j - 1) * sizeof(*top));
                top[j].idx   = i;
                top[j].score = scores[i];
                break;
            }
        }
    }
}

static char** load_class_names(const char* path) {
    char** names = calloc(MODEL_NUM_CLASSES, sizeof(char*));
    gchar* text  = NULL;
    if (!names)
        return NULL;
    if (g_file_get_contents(path, &text, NULL, NULL)) {
        gchar** lines = g_strsplit(text, "\n", -1);
        for (int i = 1; lines[i]; i++) { /* skip header: index,mid,display_name */
            gchar** f = g_strsplit(lines[i], ",", 3);
            if (f[0] && f[1] && f[2]) {
                int idx = atoi(f[0]);
                if (idx >= 0 && idx < MODEL_NUM_CLASSES && !names[idx]) {
                    gchar* n = g_strstrip(g_strdup(f[2]));
                    size_t l = strlen(n);
                    if (l >= 2 && n[0] == '"' && n[l - 1] == '"') { /* "Railroad car, train wagon" */
                        n[l - 1] = '\0';
                        memmove(n, n + 1, l - 1);
                    }
                    names[idx] = n;
                }
            }
            g_strfreev(f);
        }
        g_strfreev(lines);
        g_free(text);
    } else {
        syslog(LOG_WARNING, "[traindet] could not read %s", path);
    }
    for (int i = 0; i < MODEL_NUM_CLASSES; i++)
        if (!names[i])
            names[i] = g_strdup_printf("class_%d", i);
    return names;
}

/* ---------- parameters ---------- */

static double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Applies one parameter; caller holds g_app.lock. */
static void apply_param(det_config_t* c, const char* name, const char* value) {
    if (!value)
        return;
    if (strcmp(name, "Enabled") && strcmp(name, "HornTriggers") && strcmp(name, "ScoreSource") &&
        !isfinite(atof(value)))
        return; /* keep the previous value: NaN would defeat the clamping */
    if (!strcmp(name, "Enabled"))
        c->enabled = !strcmp(value, "yes");
    else if (!strcmp(name, "Sensitivity"))
        c->sensitivity = (int)clampd(atof(value), 1, 100);
    else if (!strcmp(name, "TriggerSeconds"))
        c->trigger_s = clampd(atof(value), 0.5, 30);
    else if (!strcmp(name, "ReleaseSeconds"))
        c->release_s = clampd(atof(value), 1, 120);
    else if (!strcmp(name, "SmoothingSeconds"))
        c->smoothing_s = clampd(atof(value), 0.5, 10);
    else if (!strcmp(name, "MinLevelDbfs"))
        c->min_level_db = clampd(atof(value), -90, -10);
    else if (!strcmp(name, "HornSensitivity"))
        c->horn_sensitivity = (int)clampd(atof(value), 1, 100);
    else if (!strcmp(name, "HornTriggers"))
        c->horn_triggers = !strcmp(value, "yes");
    else if (!strcmp(name, "CooldownSeconds"))
        c->cooldown_s = clampd(atof(value), 0, 300);
    else if (!strcmp(name, "ScoreSource"))
        c->score_source = !strcmp(value, "yamnet") ? SCORE_YAMNET : (!strcmp(value, "max") ? SCORE_MAX : SCORE_HEAD);
}

static const char* PARAMS[] = {"Enabled",         "Sensitivity",  "TriggerSeconds",  "ReleaseSeconds",
                               "SmoothingSeconds", "MinLevelDbfs", "HornSensitivity", "HornTriggers",
                               "CooldownSeconds", "ScoreSource",  NULL};

static void on_param_changed(const gchar* name, const gchar* value, gpointer user_data) {
    (void)user_data;
    const char* prefix = "root." APP_NAME ".";
    const char* shortn = name;
    /* The stored group name may differ in case (e.g. root.Train_detect.*) */
    if (g_ascii_strncasecmp(name, prefix, strlen(prefix)) == 0)
        shortn = name + strlen(prefix);
    g_mutex_lock(&g_app.lock);
    apply_param(&g_app.cfg, shortn, value);
    g_mutex_unlock(&g_app.lock);
    syslog(LOG_INFO, "[traindet] param %s=%s", shortn, value ? value : "");
}

static AXParameter* setup_params(void) {
    GError* err     = NULL;
    AXParameter* ax = ax_parameter_new(APP_NAME, &err);
    if (!ax) {
        syslog(LOG_ERR, "[traindet] ax_parameter_new: %s", err ? err->message : "?");
        g_clear_error(&err);
        return NULL;
    }
    for (int i = 0; PARAMS[i]; i++) {
        gchar* v = NULL;
        if (ax_parameter_get(ax, PARAMS[i], &v, &err)) {
            g_mutex_lock(&g_app.lock);
            apply_param(&g_app.cfg, PARAMS[i], v);
            g_mutex_unlock(&g_app.lock);
            g_free(v);
        } else {
            syslog(LOG_WARNING, "[traindet] param %s: %s", PARAMS[i], err ? err->message : "?");
            g_clear_error(&err);
        }
        if (!ax_parameter_register_callback(ax, PARAMS[i], on_param_changed, NULL, &err)) {
            syslog(LOG_WARNING, "[traindet] register callback %s: %s", PARAMS[i], err ? err->message : "?");
            g_clear_error(&err);
        }
    }
    return ax;
}

/* ---------- events ---------- */

typedef struct {
    gboolean active;
    gboolean horn;
    gdouble confidence;
} event_msg_t;

static void declaration_complete(guint declaration, gpointer user_data) {
    (void)user_data;
    event_declared = TRUE;
    syslog(LOG_INFO, "[traindet] event declaration %u complete", declaration);
}

static void declare_event(void) {
    GError* err             = NULL;
    AXEventKeyValueSet* kvs = ax_event_key_value_set_new();
    gboolean active         = FALSE;
    gboolean horn           = FALSE;
    gdouble conf            = 0.0;
    ax_event_key_value_set_add_key_value(kvs, "topic0", "tnsaxis", "CameraApplicationPlatform",
                                         AX_VALUE_TYPE_STRING, NULL);
    ax_event_key_value_set_add_key_value(kvs, "topic1", "tnsaxis", "TrainDetect", AX_VALUE_TYPE_STRING, NULL);
    ax_event_key_value_set_add_nice_names(kvs, "topic1", "tnsaxis", "Train Audio Detection",
                                          "Train Audio Detection", NULL);
    ax_event_key_value_set_add_key_value(kvs, "topic2", "tnsaxis", "TrainDetected", AX_VALUE_TYPE_STRING, NULL);
    ax_event_key_value_set_add_nice_names(kvs, "topic2", "tnsaxis", "Train detected", "Train detected", NULL);
    ax_event_key_value_set_add_key_value(kvs, "active", NULL, &active, AX_VALUE_TYPE_BOOL, NULL);
    ax_event_key_value_set_mark_as_data(kvs, "active", NULL, NULL);
    ax_event_key_value_set_add_nice_names(kvs, "active", NULL, "Active", NULL, NULL);
    ax_event_key_value_set_add_key_value(kvs, "horn", NULL, &horn, AX_VALUE_TYPE_BOOL, NULL);
    ax_event_key_value_set_mark_as_data(kvs, "horn", NULL, NULL);
    ax_event_key_value_set_add_nice_names(kvs, "horn", NULL, "Horn heard", NULL, NULL);
    ax_event_key_value_set_add_key_value(kvs, "confidence", NULL, &conf, AX_VALUE_TYPE_DOUBLE, NULL);
    ax_event_key_value_set_mark_as_data(kvs, "confidence", NULL, NULL);
    ax_event_key_value_set_mark_as_user_defined(kvs, "confidence", NULL, "wstype:xs:float", NULL);
    if (!ax_event_handler_declare2(event_handler, kvs, FALSE, (gchar*)"active", &event_decl,
                                   (AXDeclarationCompleteCallback)declaration_complete, NULL, &err)) {
        syslog(LOG_ERR, "[traindet] event declare failed: %s", err ? err->message : "?");
        g_clear_error(&err);
    }
    ax_event_key_value_set_free(kvs);
}

static gboolean send_event_main(gpointer data) {
    event_msg_t* m = data;
    if (event_handler && event_declared) {
        GError* err             = NULL;
        AXEventKeyValueSet* kvs = ax_event_key_value_set_new();
        ax_event_key_value_set_add_key_value(kvs, "active", NULL, &m->active, AX_VALUE_TYPE_BOOL, NULL);
        ax_event_key_value_set_add_key_value(kvs, "horn", NULL, &m->horn, AX_VALUE_TYPE_BOOL, NULL);
        ax_event_key_value_set_add_key_value(kvs, "confidence", NULL, &m->confidence, AX_VALUE_TYPE_DOUBLE, NULL);
        AXEvent* ev = ax_event_new2(kvs, NULL);
        ax_event_key_value_set_free(kvs);
        if (!ax_event_handler_send_event(event_handler, event_decl, ev, &err)) {
            syslog(LOG_WARNING, "[traindet] send event failed: %s", err ? err->message : "?");
            g_clear_error(&err);
        }
        ax_event_free(ev);
    }
    g_free(m);
    return G_SOURCE_REMOVE;
}

static void queue_event(gboolean active, gboolean horn, double conf) {
    event_msg_t* m = g_new0(event_msg_t, 1);
    m->active      = active;
    m->horn        = horn;
    m->confidence  = conf;
    g_idle_add(send_event_main, m); /* always runs on the main loop thread */
}

/* ---------- inference loop ---------- */

static gpointer inference_thread(gpointer data) {
    (void)data;
    static float window[WINDOW_SAMPLES];
    static model_output_t out;
    uint64_t cursor     = 0;
    double last_err_log = 0;
    double last_audio   = mono_s();
    char err[160];

    while (!g_atomic_int_get(&stop_flag)) {
        uint64_t prev = cursor;
        if (!audio_wait_latest(g_app.audio, &cursor, HOP_SAMPLES, window, WINDOW_SAMPLES, 2000)) {
            /* status shows streaming=false meanwhile. If capture never (re)starts, exit
             * so that runMode=respawn restarts the app and its PipeWire setup. */
            if (mono_s() - last_audio > AUDIO_WATCHDOG_S) {
                syslog(LOG_ERR, "[traindet] no audio for %d s, exiting for respawn", AUDIO_WATCHDOG_S);
                exit(EXIT_FAILURE);
            }
            continue;
        }
        last_audio = mono_s();
        uint64_t advanced = prev ? cursor - prev : HOP_SAMPLES;
        double dt         = (double)advanced / AUDIO_RATE;
        if (dt > 5.0)
            dt = 5.0;

        bool ok = model_run(g_app.model, window, &out, err, sizeof(err));
        float level = app_level_dbfs(window, WINDOW_SAMPLES);
        double now  = mono_s();
        double wall = wall_s();

        g_mutex_lock(&g_app.lock);
        if (advanced > (uint64_t)HOP_SAMPLES * 3 / 2)
            g_app.skipped_windows += advanced / HOP_SAMPLES - 1;
        if (!ok) {
            g_app.inference_errors++;
            snprintf(g_app.last_error, sizeof(g_app.last_error), "%s", err);
            g_mutex_unlock(&g_app.lock);
            if (now - last_err_log > 30) {
                syslog(LOG_ERR, "[traindet] inference failed: %s", err);
                last_err_log = now;
            }
            g_usleep(200000);
            continue;
        }
        float rail     = app_max_of(out.scores, RAIL_IDX, RAIL_IDX_N);
        float horn     = app_max_of(out.scores, HORN_IDX, HORN_IDX_N);
        det_result_t r = detector_update(&g_app.det, &g_app.cfg, now, wall, dt, out.train_prob, rail, horn, level);
        g_app.last          = r;
        g_app.last_head     = out.train_prob;
        g_app.last_rail     = rail;
        g_app.last_horn     = horn;
        g_app.last_level_db = level;
        app_top_classes(out.scores, g_app.top, TOP_K);
        g_app.inferences++;
        g_app.latency_ms_avg = g_app.inferences == 1 ? out.latency_ms : 0.95 * g_app.latency_ms_avg + 0.05 * out.latency_ms;
        if (out.latency_ms > g_app.latency_ms_max)
            g_app.latency_ms_max = out.latency_ms;
        g_app.last_infer_wall = wall;
        det_event_t cur = g_app.det.cur;
        det_event_t ended_copy;
        memset(&ended_copy, 0, sizeof(ended_copy));
        const det_event_t* ended = r.ended ? detector_history(&g_app.det, 0) : NULL;
        if (ended)
            ended_copy = *ended;
        g_mutex_unlock(&g_app.lock);

        if (r.started) {
            syslog(LOG_INFO, "[traindet] TRAIN START score=%.2f thr=%.2f level=%.1fdBFS horn=%d", r.score_smooth,
                   r.threshold, level, cur.horn);
            queue_event(TRUE, cur.horn, r.score_smooth);
        }
        if (r.horn_now) {
            syslog(LOG_INFO, "[traindet] horn score=%.2f thr=%.2f", horn, r.horn_threshold);
            if (r.state == DET_ACTIVE && !r.started)
                queue_event(TRUE, TRUE, r.score_smooth); /* update horn flag on the active event */
        }
        if (ended) {
            syslog(LOG_INFO, "[traindet] TRAIN END duration=%.1fs peak=%.2f mean=%.2f peak_level=%.1fdBFS horn=%d",
                   ended_copy.duration_s, ended_copy.peak_score, ended_copy.mean_score, ended_copy.peak_level_db,
                   ended_copy.horn);
            queue_event(FALSE, ended_copy.horn, ended_copy.peak_score);
        }
    }
    return NULL;
}

/* ---------- main ---------- */

static gboolean on_unix_signal(gpointer data) {
    (void)data;
    g_main_loop_quit(main_loop);
    return G_SOURCE_REMOVE;
}

int main(int argc, char** argv) {
    char err[256];
    openlog(APP_NAME, LOG_PID, LOG_USER);
    setenv("PIPEWIRE_DEBUG", "2", 0); /* pipewire warnings/errors on stderr (journal via ACAP runtime) */
    syslog(LOG_INFO, "[traindet] starting");

    memset(&g_app, 0, sizeof(g_app));
    g_mutex_init(&g_app.lock);
    detector_config_defaults(&g_app.cfg);
    detector_init(&g_app.det);
    for (int i = 0; i < TOP_K; i++)
        g_app.top[i].idx = -1;
    g_app.start_mono  = mono_s();
    g_app.class_names = load_class_names(CLASSMAP_PATH);

    main_loop = g_main_loop_new(NULL, FALSE);
    g_unix_signal_add(SIGTERM, on_unix_signal, NULL);
    g_unix_signal_add(SIGINT, on_unix_signal, NULL);

    AXParameter* params = setup_params();

    g_app.model = model_new(MODEL_PATH, LAROD_DEVICES, err, sizeof(err));
    if (!g_app.model) {
        syslog(LOG_ERR, "[traindet] model load failed: %s", err);
        return EXIT_FAILURE; /* respawn retries */
    }
    g_app.audio = audio_start(&argc, &argv);
    if (!g_app.audio) {
        syslog(LOG_ERR, "[traindet] audio start failed");
        return EXIT_FAILURE;
    }

    event_handler = ax_event_handler_new();
    declare_event();

    if (!web_start())
        syslog(LOG_ERR, "[traindet] web API failed to start (status/analyze unavailable)");

    GThread* infer = g_thread_new("traindet-infer", inference_thread, NULL);
    syslog(LOG_INFO, "[traindet] running (device %s)", model_device(g_app.model));

    g_main_loop_run(main_loop);

    syslog(LOG_INFO, "[traindet] stopping");
    g_atomic_int_set(&stop_flag, 1);
    g_thread_join(infer);
    if (event_declared)
        ax_event_handler_undeclare(event_handler, event_decl, NULL);
    ax_event_handler_free(event_handler);
    event_handler = NULL;
    /* Audio and model are not freed: FastCGI workers may still be serving a request and
     * the process is exiting anyway (larod releases private models on disconnect). */
    if (params)
        ax_parameter_free(params);
    g_main_loop_unref(main_loop);
    return EXIT_SUCCESS;
}
