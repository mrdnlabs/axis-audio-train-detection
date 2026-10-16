/**
 * train_detect - on-device audio train detection for Axis devices.
 *
 * Threads:
 *   main      GLib main loop: AXParameter callbacks and AXEvent declaration/sending.
 *   audio     PipeWire thread loop: microphone -> 16 kHz ring buffer (audio.c).
 *   inference every 0.48 s: last 0.975 s -> larod (YAMNet + train head) -> detector.
 *   web       FastCGI workers: status.cgi, timeline.cgi, analyze.cgi, label.cgi (web.c).
 *
 * Failure policy: if inference keeps failing, or audio that was streaming stops for
 * AUDIO_WATCHDOG_S, any active event is ended (active=false is sent) and the app exits
 * non-zero from the main loop so runMode=respawn restarts it. A device that never
 * delivered audio is reported in status.cgi instead of crash-looping.
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
#include <unistd.h>

#include "app.h"

#define MODEL_PATH "/usr/local/packages/" APP_NAME "/model/traindet.tflite"
#define CLASSMAP_PATH "/usr/local/packages/" APP_NAME "/model/yamnet_class_map.csv"
#define LAROD_DEVICES "cpu-tflite"
#define AUDIO_WATCHDOG_S 60
#define MAX_CONSECUTIVE_INFER_ERRORS 25 /* ~10 s */
#define STARTUP_FAIL_BACKOFF_S 30

app_t g_app;

const int RAIL_IDX[]  = {322, 323, 324, 325, 326, 327, 328};
const int RAIL_IDX_N  = 7;
const int HORN_IDX[]  = {324, 325};
const int HORN_IDX_N  = 2;

/* YAMNet class indices per timeline group (yamnet_class_map.csv). */
static const int G_RAIL_IDX[]       = {322, 323, 326, 327, 328};
static const int G_OTHER_HORN_IDX[] = {302, 312, 395, 397}; /* vehicle/air horn, foghorn, steam whistle */
static const int G_VEHICLE_IDX[]    = {294, 300, 301, 308, 310, 315, 321, 337, 343, 346};
static const int G_SPEECH_IDX[]     = {0, 1, 2, 3, 4, 5};
static const int G_MUSIC_IDX[]      = {132};
const char* const GROUP_NAMES[G_COUNT] = {"rail", "train_horn", "other_horn", "vehicle", "speech", "music"};

void app_groups(const float* scores, float* groups) {
#define N(a) ((int)(sizeof(a) / sizeof((a)[0])))
    groups[G_RAIL]       = app_max_of(scores, G_RAIL_IDX, N(G_RAIL_IDX));
    groups[G_TRAIN_HORN] = app_max_of(scores, HORN_IDX, HORN_IDX_N);
    groups[G_OTHER_HORN] = app_max_of(scores, G_OTHER_HORN_IDX, N(G_OTHER_HORN_IDX));
    groups[G_VEHICLE]    = app_max_of(scores, G_VEHICLE_IDX, N(G_VEHICLE_IDX));
    groups[G_SPEECH]     = app_max_of(scores, G_SPEECH_IDX, N(G_SPEECH_IDX));
    groups[G_MUSIC]      = app_max_of(scores, G_MUSIC_IDX, N(G_MUSIC_IDX));
#undef N
}

static GMainLoop* main_loop;
static AXEventHandler* event_handler;
/* Three AXEvents (topics under CameraApplicationPlatform/TrainDetect):
 *   TrainPresent   stateful property event, data = `active` only. Rules: "while active"
 *                  conditions or once-per-train triggers.
 *                  Never change a declared topic's data keys: the event system keeps every
 *                  schema it has seen for a topic until reboot, and conflicting schemas make
 *                  AddActionRule fail with "expression matched multiple property events"
 *                  (that is why this topic replaced v0.1-0.5.0's TrainDetected).
 *   HornDetected   stateless: a horn/whistle was heard (confidence, test).
 *   TrainPassed    stateless summary when a pass ends (duration, peak confidence, horn, test). */
enum { EV_STATE, EV_HORN, EV_PASSED, EV_COUNT };
static guint event_decl[EV_COUNT];
static gboolean event_declared[EV_COUNT];
static volatile gint stop_flag;
static int exit_code = EXIT_SUCCESS;

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

/* Strict number parse: whole string, finite. */
static bool parse_num(const char* value, double* out) {
    char* end = NULL;
    if (!value || !*value)
        return false;
    double v = g_ascii_strtod(value, &end);
    while (end && g_ascii_isspace(*end))
        end++;
    if (!end || *end != '\0' || !isfinite(v))
        return false;
    *out = v;
    return true;
}

/* Applies one parameter; caller holds g_app.lock. Invalid numbers keep the previous
 * value. Returns the effective value of the free-form numeric (string-typed) params
 * via *effective so the caller can write a corrected value back to the store. */
static bool apply_param(det_config_t* c, const char* name, const char* value, double* effective) {
    double v = 0.0;
    bool ok;
    if (!value)
        return false;
    if (!strcmp(name, "Enabled")) {
        c->enabled = !strcmp(value, "yes");
        return true;
    }
    if (!strcmp(name, "HornTriggers")) {
        c->horn_triggers = !strcmp(value, "yes");
        return true;
    }
    if (!strcmp(name, "RequireRailSound")) {
        g_app.fcfg.require_rail = !strcmp(value, "yes");
        return true;
    }
    if (!strcmp(name, "HornIncludesOtherHorns")) {
        g_app.fcfg.other_horns = !strcmp(value, "yes");
        return true;
    }
    if (!strcmp(name, "ScoreSource")) {
        c->score_source = !strcmp(value, "yamnet") ? SCORE_YAMNET : (!strcmp(value, "max") ? SCORE_MAX : SCORE_HEAD);
        return true;
    }
    ok = parse_num(value, &v);
    if (!strcmp(name, "Sensitivity")) {
        if (ok)
            c->sensitivity = (int)clampd(v, 1, 100);
    } else if (!strcmp(name, "TriggerSeconds")) {
        if (ok)
            c->trigger_s = clampd(v, 0.5, 30);
        *effective = c->trigger_s;
    } else if (!strcmp(name, "ReleaseSeconds")) {
        if (ok)
            c->release_s = clampd(v, 1, 120);
        *effective = c->release_s;
    } else if (!strcmp(name, "SmoothingSeconds")) {
        if (ok)
            c->smoothing_s = clampd(v, 0.5, 10);
        *effective = c->smoothing_s;
    } else if (!strcmp(name, "MinLevelDbfs")) {
        if (ok)
            c->min_level_db = clampd(v, -90, -10);
    } else if (!strcmp(name, "HornSensitivity")) {
        if (ok)
            c->horn_sensitivity = (int)clampd(v, 1, 100);
    } else if (!strcmp(name, "CooldownSeconds")) {
        if (ok)
            c->cooldown_s = clampd(v, 0, 300);
    } else if (!strcmp(name, "MaxEventSeconds")) {
        if (ok)
            c->max_event_s = clampd(v, 0, 3600);
    } else if (!strcmp(name, "SpeechSuppression")) {
        if (ok)
            g_app.fcfg.speech_suppression = (float)clampd(v, 0, 100) / 100.0f;
    } else if (!strcmp(name, "MinRiseDb")) {
        if (ok)
            g_app.fcfg.min_rise_db = (float)clampd(v, 0, 20);
    }
    return ok;
}

static bool is_free_numeric(const char* name) {
    return !strcmp(name, "TriggerSeconds") || !strcmp(name, "ReleaseSeconds") || !strcmp(name, "SmoothingSeconds");
}

typedef struct {
    gchar* name;
    gchar* value; /* corrected value to store */
    gchar* bad;   /* the stored value being corrected */
} param_fix_t;

static GAsyncQueue* fix_queue;
static AXParameter* g_params; /* the app's only handle: AXParameter allows one per app */

/* Corrections are written from this worker thread (sharing the app's handle; a second
 * handle aborts with "Unable to register objects on DBus").
 * Calling ax_parameter_set() on the main loop can deadlock against the parameter
 * daemon: it blocks waiting for parhand while parhand waits for the main loop to
 * accept the next change callback (seen on-device: a 25 s D-Bus timeout under rapid
 * param.cgi updates). Here the main loop stays free to answer parhand. */
static gpointer write_back_thread(gpointer data) {
    AXParameter* ax = data;
    GError* err     = NULL;
    for (;;) {
        param_fix_t* f = g_async_queue_pop(fix_queue);
        g_usleep(300000); /* let a burst of param.cgi updates settle */
        /* Only correct if the store still holds the bad value: a newer (valid) update
         * that arrived meanwhile must not be overwritten by a stale correction. */
        gchar* cur = NULL;
        if (ax_parameter_get(ax, f->name, &cur, &err) && g_strcmp0(cur, f->bad) == 0) {
            if (!ax_parameter_set(ax, f->name, f->value, TRUE, &err)) {
                syslog(LOG_WARNING, "[traindet] could not correct %s: %s", f->name, err ? err->message : "?");
                g_clear_error(&err);
            }
        }
        g_clear_error(&err);
        g_free(cur);
        g_free(f->bad);
        g_free(f->name);
        g_free(f->value);
        g_free(f);
    }
    return NULL;
}

/* The int/bool/enum params are validated by the parameter store itself; the
 * string-typed seconds params are not, so store the effective value instead of
 * leaving e.g. "nan" or "1e308" in the store while the app uses something else. */
static void maybe_write_back(const char* name, const char* value, bool parsed, double effective) {
    char buf[G_ASCII_DTOSTR_BUF_SIZE];
    double stored = 0.0;
    if (!is_free_numeric(name))
        return;
    if (parsed && parse_num(value, &stored) && fabs(stored - effective) < 1e-9)
        return;
    g_ascii_formatd(buf, sizeof(buf), "%g", effective);
    param_fix_t* f = g_new0(param_fix_t, 1);
    f->name        = g_strdup(name);
    f->value       = g_strdup(buf);
    f->bad         = g_strdup(value ? value : "");
    syslog(LOG_WARNING, "[traindet] param %s=%s invalid or out of range, storing %s", name, value ? value : "", buf);
    g_async_queue_push(fix_queue, f);
}

static const char* PARAMS[] = {"Enabled",         "Sensitivity",  "TriggerSeconds",  "ReleaseSeconds",
                               "SmoothingSeconds", "MinLevelDbfs", "HornSensitivity", "HornTriggers",
                               "CooldownSeconds", "ScoreSource",  "MaxEventSeconds", "SpeechSuppression",
                               "RequireRailSound", "MinRiseDb", "HornIncludesOtherHorns", NULL};

static void on_param_changed(const gchar* name, const gchar* value, gpointer user_data) {
    (void)user_data;
    const char* prefix = "root." APP_NAME ".";
    const char* shortn = name;
    /* The stored group name may differ in case (e.g. root.Train_detect.*) */
    if (g_ascii_strncasecmp(name, prefix, strlen(prefix)) == 0)
        shortn = name + strlen(prefix);
    double effective = 0.0;
    g_mutex_lock(&g_app.lock);
    bool parsed = apply_param(&g_app.cfg, shortn, value, &effective);
    g_mutex_unlock(&g_app.lock);
    syslog(LOG_INFO, "[traindet] param %s=%s", shortn, value ? value : "");
    maybe_write_back(shortn, value, parsed, effective);
}

static AXParameter* setup_params(void) {
    GError* err     = NULL;
    AXParameter* ax = ax_parameter_new(APP_NAME, &err);
    if (!ax) {
        syslog(LOG_ERR, "[traindet] ax_parameter_new: %s", err ? err->message : "?");
        g_clear_error(&err);
        return NULL;
    }
    g_params  = ax;
    fix_queue = g_async_queue_new();
    g_thread_unref(g_thread_new("traindet-params", write_back_thread, ax));
    for (int i = 0; PARAMS[i]; i++) {
        gchar* v = NULL;
        if (ax_parameter_get(ax, PARAMS[i], &v, &err)) {
            double effective = 0.0;
            g_mutex_lock(&g_app.lock);
            bool parsed = apply_param(&g_app.cfg, PARAMS[i], v, &effective);
            g_mutex_unlock(&g_app.lock);
            maybe_write_back(PARAMS[i], v, parsed, effective);
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
    int kind;
    gboolean active;
    gboolean horn;
    gboolean test;
    gdouble confidence;
    gdouble duration;
} event_msg_t;

static void declaration_complete(guint declaration, gpointer user_data) {
    int kind              = GPOINTER_TO_INT(user_data);
    event_declared[kind]  = TRUE;
    gboolean all          = event_declared[EV_STATE] && event_declared[EV_HORN] && event_declared[EV_PASSED];
    g_atomic_int_set(&g_app.events_declared, all ? 1 : 0);
    syslog(LOG_INFO, "[traindet] event declaration %u (kind %d) complete", declaration, kind);
}

static AXEventKeyValueSet* topic_kvs(const char* topic2, const char* nice) {
    AXEventKeyValueSet* kvs = ax_event_key_value_set_new();
    ax_event_key_value_set_add_key_value(kvs, "topic0", "tnsaxis", "CameraApplicationPlatform",
                                         AX_VALUE_TYPE_STRING, NULL);
    ax_event_key_value_set_add_key_value(kvs, "topic1", "tnsaxis", "TrainDetect", AX_VALUE_TYPE_STRING, NULL);
    ax_event_key_value_set_add_nice_names(kvs, "topic1", "tnsaxis", "Train Audio Detection",
                                          "Train Audio Detection", NULL);
    ax_event_key_value_set_add_key_value(kvs, "topic2", "tnsaxis", topic2, AX_VALUE_TYPE_STRING, NULL);
    ax_event_key_value_set_add_nice_names(kvs, "topic2", "tnsaxis", nice, nice, NULL);
    return kvs;
}

static void add_data(AXEventKeyValueSet* kvs, const char* key, AXEventValueType type, gconstpointer v,
                     const char* nice, const char* wstype) {
    ax_event_key_value_set_add_key_value(kvs, key, NULL, v, type, NULL);
    ax_event_key_value_set_mark_as_data(kvs, key, NULL, NULL);
    if (nice)
        ax_event_key_value_set_add_nice_names(kvs, key, NULL, nice, NULL, NULL);
    if (wstype)
        ax_event_key_value_set_mark_as_user_defined(kvs, key, NULL, wstype, NULL);
}

static void declare(int kind, AXEventKeyValueSet* kvs, gboolean stateless, const char* property) {
    GError* err = NULL;
    if (!ax_event_handler_declare2(event_handler, kvs, stateless, (gchar*)property, &event_decl[kind],
                                   (AXDeclarationCompleteCallback)declaration_complete, GINT_TO_POINTER(kind),
                                   &err)) {
        syslog(LOG_ERR, "[traindet] event declare (kind %d) failed: %s", kind, err ? err->message : "?");
        g_clear_error(&err);
    }
    ax_event_key_value_set_free(kvs);
}

static void declare_events(void) {
    gboolean f = FALSE;
    gdouble z  = 0.0;
    AXEventKeyValueSet* k;

    k = topic_kvs("TrainPresent", "Train present");
    add_data(k, "active", AX_VALUE_TYPE_BOOL, &f, "Active", NULL);
    declare(EV_STATE, k, FALSE, "active");

    k = topic_kvs("HornDetected", "Horn detected");
    add_data(k, "confidence", AX_VALUE_TYPE_DOUBLE, &z, "Confidence", "wstype:xs:float");
    add_data(k, "test", AX_VALUE_TYPE_BOOL, &f, "Test audio (injected)", NULL);
    declare(EV_HORN, k, TRUE, NULL);

    k = topic_kvs("TrainPassed", "Train passed");
    add_data(k, "duration", AX_VALUE_TYPE_DOUBLE, &z, "Duration (s)", "wstype:xs:float");
    add_data(k, "confidence", AX_VALUE_TYPE_DOUBLE, &z, "Peak confidence", "wstype:xs:float");
    add_data(k, "horn", AX_VALUE_TYPE_BOOL, &f, "Horn heard", NULL);
    add_data(k, "test", AX_VALUE_TYPE_BOOL, &f, "Test audio (injected)", NULL);
    declare(EV_PASSED, k, TRUE, NULL);
}

static gboolean send_event_main(gpointer data) {
    event_msg_t* m = data;
    if (event_handler && event_declared[m->kind]) {
        GError* err             = NULL;
        AXEventKeyValueSet* kvs = ax_event_key_value_set_new();
        if (m->kind == EV_STATE) {
            ax_event_key_value_set_add_key_value(kvs, "active", NULL, &m->active, AX_VALUE_TYPE_BOOL, NULL);
        } else if (m->kind == EV_HORN) {
            ax_event_key_value_set_add_key_value(kvs, "confidence", NULL, &m->confidence, AX_VALUE_TYPE_DOUBLE, NULL);
            ax_event_key_value_set_add_key_value(kvs, "test", NULL, &m->test, AX_VALUE_TYPE_BOOL, NULL);
        } else {
            ax_event_key_value_set_add_key_value(kvs, "duration", NULL, &m->duration, AX_VALUE_TYPE_DOUBLE, NULL);
            ax_event_key_value_set_add_key_value(kvs, "confidence", NULL, &m->confidence, AX_VALUE_TYPE_DOUBLE, NULL);
            ax_event_key_value_set_add_key_value(kvs, "horn", NULL, &m->horn, AX_VALUE_TYPE_BOOL, NULL);
            ax_event_key_value_set_add_key_value(kvs, "test", NULL, &m->test, AX_VALUE_TYPE_BOOL, NULL);
        }
        AXEvent* ev = ax_event_new2(kvs, NULL);
        ax_event_key_value_set_free(kvs);
        if (!ax_event_handler_send_event(event_handler, event_decl[m->kind], ev, &err)) {
            syslog(LOG_WARNING, "[traindet] send event failed: %s", err ? err->message : "?");
            g_clear_error(&err);
        }
        ax_event_free(ev);
    }
    g_free(m);
    return G_SOURCE_REMOVE;
}

static void queue_msg(event_msg_t m) {
    g_idle_add(send_event_main, g_memdup2(&m, sizeof(m))); /* always runs on the main loop thread */
}

static void queue_state(gboolean active) {
    queue_msg((event_msg_t){.kind = EV_STATE, .active = active});
}

static void queue_passed(const det_event_t* e) {
    queue_msg((event_msg_t){.kind       = EV_PASSED,
                            .duration   = e->duration_s,
                            .confidence = e->peak_score,
                            .horn       = e->horn,
                            .test       = e->test});
}

static gboolean quit_main(gpointer data) {
    (void)data;
    g_main_loop_quit(main_loop);
    return G_SOURCE_REMOVE;
}

/* Called from the inference thread: end any active event (so rules see active=false),
 * then leave the main loop with a failure code so respawn restarts the app. The idle
 * callbacks run in FIFO order, so the inactive event is sent before the quit. */
static void failsafe_exit(const char* reason) {
    syslog(LOG_ERR, "[traindet] %s; ending any active event and exiting for respawn", reason);
    g_mutex_lock(&g_app.lock);
    bool ended    = detector_force_end(&g_app.det, &g_app.cfg, mono_s(), wall_s());
    det_event_t e = ended ? *detector_history(&g_app.det, 0) : g_app.det.cur;
    snprintf(g_app.last_error, sizeof(g_app.last_error), "%s", reason);
    g_mutex_unlock(&g_app.lock);
    if (ended) {
        queue_state(FALSE);
        queue_passed(&e);
    }
    exit_code = EXIT_FAILURE;
    g_atomic_int_set(&stop_flag, 1);
    g_idle_add(quit_main, NULL);
}

/* ---------- inference loop ---------- */

static gpointer inference_thread(gpointer data) {
    (void)data;
    static float window[WINDOW_SAMPLES];
    static model_output_t out;
    uint64_t cursor     = 0;
    double last_err_log = 0;
    double last_audio   = mono_s();
    bool had_audio      = false;
    int consecutive_err = 0;
    char err[160];

    while (!g_atomic_int_get(&stop_flag)) {
        uint64_t prev = cursor;
        if (!audio_wait_latest(g_app.audio, &cursor, HOP_SAMPLES, window, WINDOW_SAMPLES, 2000)) {
            if (!had_audio) {
                /* Never had audio (mic disabled / no input): report, don't crash-loop. */
                g_mutex_lock(&g_app.lock);
                snprintf(g_app.last_error, sizeof(g_app.last_error),
                         "no audio from the microphone; check that audio is enabled on the device");
                g_mutex_unlock(&g_app.lock);
            } else if (mono_s() - last_audio > AUDIO_WATCHDOG_S) {
                failsafe_exit("audio stopped for 60 s");
                break;
            }
            continue;
        }
        had_audio         = true;
        last_audio        = mono_s();
        uint64_t advanced = prev ? cursor - prev : HOP_SAMPLES;
        /* Cap the time step: after a stall, one window must not count as seconds of
         * evidence toward the trigger or release timers. */
        double dt = (double)advanced / AUDIO_RATE;
        if (dt > 1.0)
            dt = 1.0;

        float horn_eff = 0.0f;
        audio_info_t ai;
        audio_get_info(g_app.audio, &ai);
        bool injecting = ai.inject_remaining > 0;
        bool ok        = model_run(g_app.model, window, &out, err, sizeof(err));
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
            if (++consecutive_err >= MAX_CONSECUTIVE_INFER_ERRORS) {
                failsafe_exit("inference keeps failing (larod)");
                break;
            }
            g_usleep(200000);
            continue;
        }
        consecutive_err = 0;
        float rail = app_max_of(out.scores, RAIL_IDX, RAIL_IDX_N);
        float horn = app_max_of(out.scores, HORN_IDX, HORN_IDX_N);
        app_groups(out.scores, g_app.last_groups);
        /* Sound rules (speech suppression, rail evidence, loudness rise, horn conjunction)
         * turn the model's score into the score the detector sees. */
        fusion_result_t fr = fusion_step(&g_app.fusion, &g_app.fcfg, out.train_prob, g_app.last_groups[G_RAIL],
                                         g_app.last_groups[G_TRAIN_HORN], g_app.last_groups[G_OTHER_HORN],
                                         g_app.last_groups[G_SPEECH], level);
        g_app.last_fusion = fr;
        horn_eff          = fr.horn;
        det_result_t r = detector_update(&g_app.det, &g_app.cfg, now, wall, dt, fr.score, rail, fr.horn, level);
        if (injecting && g_app.det.state == DET_ACTIVE)
            g_app.det.cur.test = true; /* injected audio: mark so rules/VMS can tell it apart */
        if (r.ended && injecting)
            ((det_event_t*)detector_history(&g_app.det, 0))->test = true;
        g_app.last          = r;
        g_app.last_head     = out.train_prob;
        g_app.last_rail     = rail;
        g_app.last_horn     = horn;
        g_app.last_level_db = level;
        app_top_classes(out.scores, g_app.top, TOP_K);
        {
            timeline_entry_t* te = &g_app.timeline[g_app.timeline_seq % TIMELINE_LEN];
            g_app.timeline_seq++;
            te->seq      = g_app.timeline_seq;
            te->wall     = wall;
            te->head     = out.train_prob;
            te->score    = r.score_smooth;
            te->level_db = level;
            te->state    = (uint8_t)r.state;
            memcpy(te->groups, g_app.last_groups, sizeof(te->groups));
        }
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
            queue_state(TRUE);
        }
        if (r.horn_now) {
            syslog(LOG_INFO, "[traindet] horn score=%.2f thr=%.2f", horn_eff, r.horn_threshold);
            queue_msg((event_msg_t){.kind = EV_HORN, .confidence = horn_eff, .test = injecting});
        }
        if (ended) {
            syslog(LOG_INFO, "[traindet] TRAIN END duration=%.1fs peak=%.2f mean=%.2f peak_level=%.1fdBFS horn=%d%s%s",
                   ended_copy.duration_s, ended_copy.peak_score, ended_copy.mean_score, ended_copy.peak_level_db,
                   ended_copy.horn, ended_copy.test ? " TEST" : "", ended_copy.capped ? " capped" : "");
            queue_state(FALSE);
            queue_passed(&ended_copy);
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
    fusion_config_defaults(&g_app.fcfg);
    fusion_init(&g_app.fusion);
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
        syslog(LOG_ERR, "[traindet] model load failed: %s; retrying in %d s", err, STARTUP_FAIL_BACKOFF_S);
        sleep(STARTUP_FAIL_BACKOFF_S); /* avoid a tight respawn loop */
        return EXIT_FAILURE;
    }
    g_app.audio = audio_start(&argc, &argv);
    if (!g_app.audio) {
        syslog(LOG_ERR, "[traindet] audio start failed; retrying in %d s", STARTUP_FAIL_BACKOFF_S);
        sleep(STARTUP_FAIL_BACKOFF_S);
        return EXIT_FAILURE;
    }

    event_handler = ax_event_handler_new();
    declare_events();

    if (!web_start())
        syslog(LOG_ERR, "[traindet] web API failed to start (status/analyze unavailable)");

    GThread* infer = g_thread_new("traindet-infer", inference_thread, NULL);
    syslog(LOG_INFO, "[traindet] running (device %s)", model_device(g_app.model));

    g_main_loop_run(main_loop);

    syslog(LOG_INFO, "[traindet] stopping");
    g_atomic_int_set(&stop_flag, 1);
    g_thread_join(infer);
    for (int i = 0; i < EV_COUNT; i++)
        if (event_declared[i])
            ax_event_handler_undeclare(event_handler, event_decl[i], NULL);
    ax_event_handler_free(event_handler);
    event_handler = NULL;
    /* Audio and model are not freed: FastCGI workers may still be serving a request and
     * the process is exiting anyway (larod releases private models on disconnect). */
    if (params)
        ax_parameter_free(params);
    g_main_loop_unref(main_loop);
    return exit_code;
}
