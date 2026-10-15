/* Microphone capture through PipeWire into a 16 kHz mono float ring buffer. */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define AUDIO_RATE 16000

typedef struct audio audio_t;

typedef struct {
    char node[96];
    uint32_t native_rate;
    uint32_t native_channels;
    bool streaming;
    uint64_t samples_total; /* 16 kHz samples written since start */
    double last_data_mono;  /* monotonic seconds of last buffer */
    double inject_remaining; /* seconds of injected test audio still playing */
} audio_info_t;

audio_t* audio_start(int* argc, char*** argv);
void audio_stop(audio_t* a);
/* Blocks until at least `min_new` samples beyond `*cursor` exist or `timeout_ms` passes.
 * On success copies the latest `n` samples into `dst`, sets *cursor to the stream
 * position of the newest sample and returns true. */
bool audio_wait_latest(audio_t* a, uint64_t* cursor, size_t min_new, float* dst, size_t n, int timeout_ms);
void audio_get_info(audio_t* a, audio_info_t* info);
/* Test hook: the next `n` 16 kHz samples fed to the detector come from `samples`
 * instead of the microphone (real-time pacing is kept by the mic stream). */
void audio_inject(audio_t* a, const float* samples, size_t n);
