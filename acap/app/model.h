/* Larod wrapper for the train-detection TFLite model.
 * Input: 15600 float32 samples @16 kHz. Outputs: train probability + 521 YAMNet scores. */
#pragma once

#include <stdbool.h>
#include <stddef.h>

#define MODEL_INPUT_SAMPLES 15600
#define MODEL_NUM_CLASSES 521

typedef struct model model_t;

typedef struct {
    float train_prob;
    float scores[MODEL_NUM_CLASSES];
    double latency_ms;
} model_output_t;

/* Loads `path` on the first available larod device in `devices` (comma separated). */
model_t* model_new(const char* path, const char* devices, char* err, size_t err_len);
/* Thread-safe (internally serialized). */
bool model_run(model_t* m, const float* samples, model_output_t* out, char* err, size_t err_len);
const char* model_device(const model_t* m);
void model_free(model_t* m);
