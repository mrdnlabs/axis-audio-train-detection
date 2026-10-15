#include "detector.h"

#include <math.h>
#include <string.h>

#define HORN_MIN_WINDOWS 2

static void end_event(detector_t* d, const det_config_t* c, double now_mono, double now_wall);

void detector_config_defaults(det_config_t* c) {
    c->enabled          = true;
    c->sensitivity      = 50;
    c->trigger_s        = 3.0;
    c->release_s        = 8.0;
    c->smoothing_s      = 2.0;
    c->min_level_db     = -65.0;
    c->horn_sensitivity = 50;
    c->horn_triggers    = true;
    c->cooldown_s       = 10.0;
    c->max_event_s      = 600.0;
    c->score_source     = SCORE_HEAD;
}

void detector_init(detector_t* d) {
    memset(d, 0, sizeof(*d));
    d->state = DET_IDLE;
}

static int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

float detector_threshold(int sensitivity) {
    return 0.95f - 0.009f * (float)(clampi(sensitivity, 1, 100) - 1);
}

float detector_horn_threshold(int horn_sensitivity) {
    return 0.6f - 0.0055f * (float)(clampi(horn_sensitivity, 1, 100) - 1);
}

const char* detector_state_name(det_state_t s) {
    switch (s) {
        case DET_IDLE:
            return "idle";
        case DET_CANDIDATE:
            return "candidate";
        case DET_ACTIVE:
            return "active";
    }
    return "unknown";
}

const det_event_t* detector_history(const detector_t* d, int i) {
    if (i < 0 || i >= d->hist_count)
        return NULL;
    int idx = (d->hist_head - 1 - i + DET_HISTORY_LEN) % DET_HISTORY_LEN;
    return &d->hist[idx];
}

bool detector_set_label(detector_t* d, double start_wall, int8_t label) {
    for (int i = 0; i < d->hist_count; i++) {
        det_event_t* e = &d->hist[(d->hist_head - 1 - i + DET_HISTORY_LEN) % DET_HISTORY_LEN];
        if (fabs(e->start_wall - start_wall) < 0.005) {
            e->label = label;
            return true;
        }
    }
    return false;
}

bool detector_force_end(detector_t* d, const det_config_t* c, double now_mono, double now_wall) {
    if (d->state != DET_ACTIVE)
        return false;
    d->below_s = 0.0;
    end_event(d, c, now_mono, now_wall);
    return true;
}

static float smooth_push(detector_t* d, float v, int n) {
    if (n < 1)
        n = 1;
    if (n > DET_SMOOTH_MAX)
        n = DET_SMOOTH_MAX;
    d->smooth_buf[d->smooth_pos] = v;
    d->smooth_pos                = (d->smooth_pos + 1) % DET_SMOOTH_MAX;
    if (d->smooth_fill < DET_SMOOTH_MAX)
        d->smooth_fill++;
    int count = n < d->smooth_fill ? n : d->smooth_fill;
    float sum = 0.0f;
    for (int i = 0; i < count; i++)
        sum += d->smooth_buf[(d->smooth_pos - 1 - i + DET_SMOOTH_MAX) % DET_SMOOTH_MAX];
    return sum / (float)count;
}

static void start_event(detector_t* d, double now_mono, double now_wall, double lead_s, float level_db) {
    d->state = DET_ACTIVE;
    memset(&d->cur, 0, sizeof(d->cur));
    d->cur.start_wall    = now_wall - lead_s;
    d->cur.start_mono    = now_mono - lead_s;
    d->cur.peak_level_db = level_db;
    d->cur_score_sum     = 0.0;
    d->cur_windows       = 0;
    d->below_s           = 0.0;
    d->events_total++;
}

static void end_event(detector_t* d, const det_config_t* c, double now_mono, double now_wall) {
    (void)now_wall;
    /* The train left when the score first dropped, i.e. below_s ago. Durations come
     * from the monotonic clock so a wall-clock step (NTP) cannot corrupt them. */
    d->cur.duration_s = (now_mono - d->below_s) - d->cur.start_mono;
    if (d->cur.duration_s < 0)
        d->cur.duration_s = 0;
    d->cur.end_wall = d->cur.start_wall + d->cur.duration_s;
    d->cur.mean_score         = d->cur_windows ? (float)(d->cur_score_sum / d->cur_windows) : 0.0f;
    d->hist[d->hist_head]     = d->cur;
    d->hist_head              = (d->hist_head + 1) % DET_HISTORY_LEN;
    if (d->hist_count < DET_HISTORY_LEN)
        d->hist_count++;
    d->state          = DET_IDLE;
    d->above_s        = 0.0;
    d->below_s        = 0.0;
    d->cooldown_until = now_mono + c->cooldown_s;
}

det_result_t detector_update(detector_t* d,
                             const det_config_t* c,
                             double now_mono,
                             double now_wall,
                             double dt,
                             float head,
                             float rail,
                             float horn,
                             float level_db) {
    det_result_t r;
    memset(&r, 0, sizeof(r));
    r.threshold      = detector_threshold(c->sensitivity);
    r.horn_threshold = detector_horn_threshold(c->horn_sensitivity);

    float raw;
    switch (c->score_source) {
        case SCORE_YAMNET:
            raw = rail;
            break;
        case SCORE_MAX:
            raw = head > rail ? head : rail;
            break;
        case SCORE_HEAD:
        default:
            raw = head;
            break;
    }
    r.gated = level_db < c->min_level_db;
    if (r.gated) {
        raw  = 0.0f;
        horn = 0.0f;
    }
    r.score_raw    = raw;
    r.score_smooth = smooth_push(d, raw, (int)lround(c->smoothing_s / dt));

    d->horn_run = horn >= r.horn_threshold ? d->horn_run + 1 : 0;
    r.horn_now  = d->horn_run == HORN_MIN_WINDOWS; /* fire once per horn blast */
    if (r.horn_now) {
        d->horns_total++;
        d->last_horn_wall = now_wall;
    }

    bool above = r.score_smooth >= r.threshold;

    if (!c->enabled) {
        if (d->state == DET_ACTIVE) {
            d->below_s = 0.0;
            end_event(d, c, now_mono, now_wall);
            r.ended = true;
        }
        d->state   = DET_IDLE;
        d->above_s = 0.0;
        r.state    = d->state;
        return r;
    }

    switch (d->state) {
        case DET_IDLE:
        case DET_CANDIDATE: {
            bool cooling = now_mono < d->cooldown_until;
            d->above_s   = above ? d->above_s + dt : 0.0;
            if (!cooling && (d->above_s >= c->trigger_s || (c->horn_triggers && r.horn_now))) {
                start_event(d, now_mono, now_wall, d->above_s < c->trigger_s ? d->above_s : c->trigger_s, level_db);
                r.started = true;
            } else {
                d->state = d->above_s > 0.0 ? DET_CANDIDATE : DET_IDLE;
            }
            break;
        }
        case DET_ACTIVE:
            break;
    }

    if (d->state == DET_ACTIVE) {
        d->cur_score_sum += r.score_smooth;
        d->cur_windows++;
        if (r.score_smooth > d->cur.peak_score)
            d->cur.peak_score = r.score_smooth;
        if (level_db > d->cur.peak_level_db)
            d->cur.peak_level_db = level_db;
        if (horn >= r.horn_threshold && d->horn_run >= HORN_MIN_WINDOWS)
            d->cur.horn = true;
        if (!r.started) {
            d->below_s = above ? 0.0 : d->below_s + dt;
            if (d->below_s >= c->release_s) {
                end_event(d, c, now_mono, now_wall);
                r.ended = true;
            } else if (c->max_event_s > 0 && now_mono - d->cur.start_mono >= c->max_event_s) {
                /* Persistent noise above the threshold must not latch the event forever. */
                d->below_s    = 0.0;
                d->cur.capped = true;
                end_event(d, c, now_mono, now_wall);
                r.ended = true;
            }
        }
    }
    r.state = d->state;
    return r;
}
