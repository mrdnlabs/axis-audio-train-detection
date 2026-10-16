#include "fusion.h"

#include <stdlib.h>
#include <string.h>

void fusion_config_defaults(fusion_config_t* c) {
    c->speech_suppression = 1.0f;
    c->require_rail       = false;
    c->min_rise_db        = 0.0f;
    c->other_horns        = false;
}

void fusion_init(fusion_t* f) {
    memset(f, 0, sizeof(*f));
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static int cmp_float(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

fusion_result_t fusion_step(fusion_t* f, const fusion_config_t* c, float head, float rail, float train_horn,
                            float other_horn, float speech, float level_db) {
    fusion_result_t r;
    memset(&r, 0, sizeof(r));

    /* rail evidence history (rail classes or the train horn) */
    float rail_any                 = rail > train_horn ? rail : train_horn;
    f->rail_hist[f->rail_pos]      = rail_any;
    f->rail_pos                    = (f->rail_pos + 1) % FUSION_HORN_NEAR;
    if (f->rail_fill < FUSION_HORN_NEAR)
        f->rail_fill++;
    float recent = 0.0f, near = 0.0f;
    for (int i = 0; i < f->rail_fill; i++) {
        float v = f->rail_hist[(f->rail_pos - 1 - i + FUSION_HORN_NEAR) % FUSION_HORN_NEAR];
        if (i < FUSION_RECENT && v > recent)
            recent = v;
        if (v > near)
            near = v;
    }
    r.rail_recent = recent;

    /* background level: 20th percentile of the last ~60 s (computed before adding this window) */
    float bg = level_db;
    if (f->level_fill >= 20) {
        float tmp[FUSION_BG];
        memcpy(tmp, f->level_hist, sizeof(float) * (size_t)f->level_fill);
        qsort(tmp, (size_t)f->level_fill, sizeof(float), cmp_float);
        bg = tmp[f->level_fill / 5];
    }
    r.background_db              = bg;
    f->level_hist[f->level_pos]  = level_db;
    f->level_pos                 = (f->level_pos + 1) % FUSION_BG;
    if (f->level_fill < FUSION_BG)
        f->level_fill++;

    float s = head;
    if (c->speech_suppression > 0.0f)
        s *= 1.0f - clampf(c->speech_suppression, 0, 1) * clampf(speech - recent, 0, 1);
    if (c->require_rail)
        s *= clampf(recent / FUSION_RAIL_REF, 0, 1);
    if (c->min_rise_db > 0.0f && f->level_fill > 20 && level_db < bg + c->min_rise_db) {
        s            = 0.0f;
        r.below_rise = true;
    }
    r.score = clampf(s, 0, 1);

    r.horn = train_horn;
    if (c->other_horns && near >= FUSION_RAIL_REF && other_horn > r.horn)
        r.horn = other_horn;
    return r;
}
