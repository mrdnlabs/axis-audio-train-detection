/* Sound rules applied to the model's train score before the detector (pure logic,
 * host-testable). Groups are YAMNet key-sound scores for the current window. */
#pragma once

#include <stdbool.h>

#define FUSION_RECENT 6      /* ~3 s window for "recent rail evidence" */
#define FUSION_HORN_NEAR 20  /* ~10 s either side for other-horn conjunction (look-back only) */
#define FUSION_BG 125        /* ~60 s of levels for the background estimate */
#define FUSION_RAIL_REF 0.2f /* rail evidence that fully opens the rail gate */

typedef struct {
    float speech_suppression; /* 0..1: how strongly speech without rail evidence lowers the score */
    bool require_rail;        /* scale the score by recent rail evidence */
    float min_rise_db;        /* 0 = off; else level must exceed the background by this much */
    bool other_horns;         /* other horns count as horn when rail evidence is near */
} fusion_config_t;

typedef struct {
    float rail_hist[FUSION_HORN_NEAR];
    int rail_pos, rail_fill;
    float level_hist[FUSION_BG];
    int level_pos, level_fill;
} fusion_t;

typedef struct {
    float score;       /* fused train score fed to the detector */
    float horn;        /* horn evidence fed to the detector */
    float rail_recent; /* max rail/horn evidence over the last ~3 s */
    float background_db;
    bool below_rise;   /* level did not rise enough above the background */
} fusion_result_t;

void fusion_init(fusion_t* f);
void fusion_config_defaults(fusion_config_t* c);
/* rail, train_horn, other_horn, speech: group scores of this window. */
fusion_result_t fusion_step(fusion_t* f, const fusion_config_t* c, float head, float rail, float train_horn,
                            float other_horn, float speech, float level_db);
