/* Host unit tests for the pure-logic parts of the ACAP (detector, WAV parser).
 * Build/run: tools/test_host.sh */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../app/detector.h"
#include "../app/wav.h"

static int failures;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            failures++;                                   \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
        }                                                 \
    } while (0)

#define DT 0.48

typedef struct {
    int started, ended, horns;
    double t;
} run_t;

/* Feeds `n` windows with constant scores. */
static void feed(detector_t* d, det_config_t* c, run_t* r, int n, float head, float horn, float level) {
    for (int i = 0; i < n; i++) {
        r->t += DT;
        det_result_t res = detector_update(d, c, r->t, 1e9 + r->t, DT, head, 0.0f, horn, level);
        r->started += res.started;
        r->ended += res.ended;
        r->horns += res.horn_now;
    }
}

static void test_thresholds(void) {
    CHECK(fabsf(detector_threshold(1) - 0.95f) < 1e-5, "thr(1)");
    CHECK(fabsf(detector_threshold(100) - 0.059f) < 1e-3, "thr(100)=%f", detector_threshold(100));
    CHECK(fabsf(detector_threshold(50) - 0.509f) < 1e-3, "thr(50)");
    CHECK(detector_threshold(0) == detector_threshold(1), "clamp low");
    CHECK(detector_horn_threshold(1) > detector_horn_threshold(100), "horn monotonic");
}

static void test_basic_event(void) {
    detector_t d;
    det_config_t c;
    run_t r = {0};
    detector_init(&d);
    detector_config_defaults(&c);
    c.horn_triggers = false;
    feed(&d, &c, &r, 20, 0.05f, 0, -40); /* quiet background */
    CHECK(r.started == 0, "no event on background");
    feed(&d, &c, &r, 4, 0.9f, 0, -30); /* 1.9 s - shorter than trigger (3 s) after smoothing */
    CHECK(r.started == 0, "short burst must not trigger");
    feed(&d, &c, &r, 30, 0.9f, 0, -30); /* 14 s train */
    CHECK(r.started == 1, "train triggers once (got %d)", r.started);
    CHECK(d.state == DET_ACTIVE, "active during train");
    feed(&d, &c, &r, 6, 0.05f, 0, -40); /* 2.9 s gap < release */
    CHECK(r.ended == 0, "gap shorter than release keeps event");
    feed(&d, &c, &r, 10, 0.9f, 0, -30);
    feed(&d, &c, &r, 30, 0.05f, 0, -40);
    CHECK(r.ended == 1, "event ends after release (ended=%d)", r.ended);
    CHECK(r.started == 1, "gap did not create a 2nd event");
    const det_event_t* e = detector_history(&d, 0);
    CHECK(e != NULL, "history entry");
    if (e) {
        CHECK(e->duration_s > 15 && e->duration_s < 35, "duration plausible: %.1f", e->duration_s);
        CHECK(e->peak_score > 0.85f, "peak score %.2f", e->peak_score);
    }
}

static void test_level_gate_and_disable(void) {
    detector_t d;
    det_config_t c;
    run_t r = {0};
    detector_init(&d);
    detector_config_defaults(&c);
    c.min_level_db = -50;
    feed(&d, &c, &r, 30, 0.95f, 0, -60); /* confident but too quiet */
    CHECK(r.started == 0, "level gate suppresses");
    c.enabled = false;
    feed(&d, &c, &r, 30, 0.95f, 0, -20);
    CHECK(r.started == 0, "disabled suppresses");
    c.enabled = true;
    feed(&d, &c, &r, 30, 0.95f, 0, -20);
    CHECK(r.started == 1, "re-enabled triggers");
    c.enabled = false;
    feed(&d, &c, &r, 1, 0.95f, 0, -20);
    CHECK(r.ended == 1, "disable ends active event");
}

static void test_horn_and_cooldown(void) {
    detector_t d;
    det_config_t c;
    run_t r = {0};
    detector_init(&d);
    detector_config_defaults(&c);
    feed(&d, &c, &r, 3, 0.1f, 0.9f, -25); /* 1.4 s horn, train score still low */
    CHECK(r.horns == 1, "one horn fired (got %d)", r.horns);
    CHECK(r.started == 1, "horn triggers event");
    CHECK(d.cur.horn, "event flagged horn");
    feed(&d, &c, &r, 30, 0.1f, 0, -40);
    CHECK(r.ended == 1, "horn-only event ends");
    feed(&d, &c, &r, 3, 0.1f, 0.9f, -25); /* within cooldown */
    CHECK(r.started == 1, "cooldown blocks immediate retrigger");
    feed(&d, &c, &r, 30, 0.1f, 0.0f, -40);
    feed(&d, &c, &r, 3, 0.1f, 0.9f, -25);
    CHECK(r.started == 2, "after cooldown horn triggers again (started=%d)", r.started);
    c.horn_triggers = false;
    detector_init(&d);
    r = (run_t){0};
    feed(&d, &c, &r, 3, 0.1f, 0.9f, -25);
    CHECK(r.started == 0 && r.horns == 1, "horn counted but does not trigger when disabled");
}

static void test_history_ring(void) {
    detector_t d;
    det_config_t c;
    run_t r = {0};
    detector_init(&d);
    detector_config_defaults(&c);
    c.cooldown_s = 0;
    for (int i = 0; i < DET_HISTORY_LEN + 5; i++) {
        feed(&d, &c, &r, 12, 0.9f, 0, -30);
        feed(&d, &c, &r, 25, 0.0f, 0, -40);
    }
    CHECK(d.hist_count == DET_HISTORY_LEN, "history capped (%d)", d.hist_count);
    CHECK(detector_history(&d, DET_HISTORY_LEN) == NULL, "out of range is NULL");
    CHECK(d.events_total == DET_HISTORY_LEN + 5, "events_total %llu", (unsigned long long)d.events_total);
    const det_event_t* e = detector_history(&d, 3);
    double start = e->start_wall;
    CHECK(detector_set_label(&d, start + 0.001, -1), "label by start time");
    CHECK(detector_history(&d, 3)->label == -1, "label stored");
    CHECK(detector_history(&d, 2)->label == 0, "other events untouched");
    CHECK(!detector_set_label(&d, start + 5.0, 1), "unknown start rejected");
}

static void put32(unsigned char* p, unsigned v) {
    p[0] = v & 255; p[1] = (v >> 8) & 255; p[2] = (v >> 16) & 255; p[3] = (v >> 24) & 255;
}
static void put16(unsigned char* p, unsigned v) {
    p[0] = v & 255; p[1] = (v >> 8) & 255;
}

static size_t make_wav(unsigned char* b, unsigned rate, unsigned ch, unsigned bits, unsigned n) {
    memcpy(b, "RIFF", 4); put32(b + 4, 36 + n * 2); memcpy(b + 8, "WAVE", 4);
    memcpy(b + 12, "fmt ", 4); put32(b + 16, 16); put16(b + 20, 1); put16(b + 22, ch);
    put32(b + 24, rate); put32(b + 28, rate * 2 * ch); put16(b + 32, 2 * ch); put16(b + 34, bits);
    memcpy(b + 36, "data", 4); put32(b + 40, n * 2);
    for (unsigned i = 0; i < n; i++)
        put16(b + 44 + 2 * i, (unsigned)(i == 1 ? 0x8000 : (i == 2 ? 0x4000 : 0)));
    return 44 + n * 2;
}

static void test_wav(void) {
    unsigned char b[256];
    size_t n = 0;
    const char* err = NULL;
    size_t len = make_wav(b, 16000, 1, 16, 10);
    float* s = wav_decode_pcm16_mono16k(b, len, &n, &err);
    CHECK(s && n == 10, "valid wav decodes");
    if (s) {
        CHECK(s[1] == -1.0f && fabsf(s[2] - 0.5f) < 1e-6, "sample scaling");
        free(s);
    }
    len = make_wav(b, 44100, 1, 16, 10);
    CHECK(!wav_decode_pcm16_mono16k(b, len, &n, &err), "rejects 44.1 kHz");
    len = make_wav(b, 16000, 2, 16, 10);
    CHECK(!wav_decode_pcm16_mono16k(b, len, &n, &err), "rejects stereo");
    CHECK(!wav_decode_pcm16_mono16k((const unsigned char*)"RIFF", 4, &n, &err), "rejects short");
    len = make_wav(b, 16000, 1, 16, 10);
    put32(b + 40, 0x7fffffff); /* lying data size: must clamp to buffer */
    s = wav_decode_pcm16_mono16k(b, len, &n, &err);
    CHECK(s && n == 10, "clamps oversized data chunk (n=%zu)", n);
    free(s);
    put32(b + 16, 0xfffffff0); /* absurd fmt size */
    CHECK(!wav_decode_pcm16_mono16k(b, len, &n, &err) || n <= 10, "absurd fmt size is safe");
}

int main(void) {
    test_thresholds();
    test_basic_event();
    test_level_gate_and_disable();
    test_horn_and_cooldown();
    test_history_ring();
    test_wav();
    printf(failures ? "%d FAILURES\n" : "all host tests passed\n", failures);
    return failures ? 1 : 0;
}
