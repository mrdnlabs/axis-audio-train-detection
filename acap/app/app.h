/* Shared application state between the inference thread, the FastCGI threads and
 * the GLib main loop (parameters, events). Everything below `lock` is guarded by it. */
#pragma once

#include <glib.h>
#include <stdbool.h>
#include <stdint.h>

#include "audio.h"
#include "detector.h"
#include "model.h"

#define APP_NAME "train_detect"
#define WINDOW_SAMPLES MODEL_INPUT_SAMPLES
#define HOP_SAMPLES 7680
#define TOP_K 5
#define TIMELINE_LEN 625 /* ~5 min of 0.48 s windows */

/* Key sound groups shown on the timeline (max YAMNet score within each group). */
enum { G_RAIL, G_TRAIN_HORN, G_OTHER_HORN, G_VEHICLE, G_SPEECH, G_MUSIC, G_COUNT };

typedef struct {
    uint64_t seq;
    double wall;
    float head;
    float score; /* smoothed score the detector used */
    float level_db;
    float groups[G_COUNT];
    uint8_t state; /* det_state_t */
} timeline_entry_t;

typedef struct {
    int idx;
    float score;
} class_score_t;

typedef struct {
    model_t* model;    /* immutable after start (internally locked) */
    audio_t* audio;    /* immutable after start (internally locked) */
    char** class_names; /* 521 entries, immutable after start */
    double start_mono;

    GMutex lock;
    det_config_t cfg;
    detector_t det;
    det_result_t last;
    float last_head;
    float last_rail;
    float last_horn;
    float last_level_db;
    class_score_t top[TOP_K];
    uint64_t inferences;
    uint64_t inference_errors;
    uint64_t skipped_windows;
    double latency_ms_avg;
    double latency_ms_max;
    double last_infer_wall;
    char last_error[160];
    gint events_declared; /* atomic: AXEvent declaration completed */
    float last_groups[G_COUNT];
    timeline_entry_t timeline[TIMELINE_LEN];
    uint64_t timeline_seq; /* seq of the newest entry; 0 = empty */
} app_t;

extern app_t g_app;

/* Fills top-k YAMNet classes. */
void app_top_classes(const float* scores, class_score_t* top, int k);
float app_level_dbfs(const float* x, int n);
float app_max_of(const float* scores, const int* idx, int n);

extern const int RAIL_IDX[];
extern const int RAIL_IDX_N;
extern const int HORN_IDX[];
extern const int HORN_IDX_N;
extern const char* const GROUP_NAMES[G_COUNT];
void app_groups(const float* scores, float* groups);

/* web.c */
bool web_start(void);
