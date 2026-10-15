/* Minimal, bounds-checked RIFF/WAVE parser for 16 kHz mono PCM16 (host-testable). */
#pragma once

#include <stddef.h>

/* Parses `buf` and converts samples to float in [-1, 1).
 * On success returns a malloc'ed array (caller frees) and sets *n_samples.
 * On failure returns NULL and points *err at a static message. */
float* wav_decode_pcm16_mono16k(const unsigned char* buf,
                                size_t len,
                                size_t* n_samples,
                                const char** err);
