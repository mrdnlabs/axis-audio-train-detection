/* Train detection state machine: pure logic, no I/O (host-testable). */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define DET_HISTORY_LEN 50
#define DET_SMOOTH_MAX 64

typedef enum { SCORE_HEAD = 0, SCORE_YAMNET = 1, SCORE_MAX = 2 } score_source_t;

typedef enum { DET_IDLE = 0, DET_CANDIDATE = 1, DET_ACTIVE = 2 } det_state_t;

typedef struct {
    bool enabled;
    int sensitivity;      /* 1..100 */
    double trigger_s;     /* sustained time above threshold before the event fires */
    double release_s;     /* time below threshold before the event ends */
    double smoothing_s;   /* moving-average window */
    double min_level_db;  /* loudness gate, dBFS */
    int horn_sensitivity; /* 1..100 */
    bool horn_triggers;   /* a detected horn starts an event immediately */
    double cooldown_s;    /* minimum gap between events */
    score_source_t score_source;
} det_config_t;

typedef struct {
    double start_wall; /* unix seconds */
    double end_wall;
    double duration_s;
    float peak_score;
    float mean_score;
    float peak_level_db;
    bool horn;
    int8_t label; /* operator feedback: 0 not reviewed, 1 train, -1 false alarm */
} det_event_t;

typedef struct {
    det_state_t state;
    float smooth_buf[DET_SMOOTH_MAX];
    int smooth_pos;
    int smooth_fill;
    double above_s;
    double below_s;
    int horn_run;
    double cooldown_until; /* monotonic seconds */
    double last_horn_wall;
    /* current event */
    det_event_t cur;
    double cur_score_sum;
    int cur_windows;
    /* history (ring, newest at hist_head-1) */
    det_event_t hist[DET_HISTORY_LEN];
    int hist_head;
    int hist_count;
    uint64_t events_total;
    uint64_t horns_total;
} detector_t;

typedef struct {
    float score_raw;
    float score_smooth;
    float threshold;
    float horn_threshold;
    bool gated;     /* below loudness gate */
    bool horn_now;  /* horn sub-detector fired on this window */
    bool started;   /* event started on this window */
    bool ended;     /* event ended on this window */
    det_state_t state;
} det_result_t;

void detector_init(detector_t* d);
void detector_config_defaults(det_config_t* c);
float detector_threshold(int sensitivity);
float detector_horn_threshold(int horn_sensitivity);
det_result_t detector_update(detector_t* d,
                             const det_config_t* c,
                             double now_mono,
                             double now_wall,
                             double dt,
                             float head,
                             float rail,
                             float horn,
                             float level_db);
const char* detector_state_name(det_state_t s);
/* i = 0 is the newest event. */
const det_event_t* detector_history(const detector_t* d, int i);
/* Sets the operator label of the history event that started at `start_wall`
 * (matched within 5 ms). Returns false if no such event is kept. */
bool detector_set_label(detector_t* d, double start_wall, int8_t label);
