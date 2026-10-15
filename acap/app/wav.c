#include "wav.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const unsigned char* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t rd16(const unsigned char* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

float* wav_decode_pcm16_mono16k(const unsigned char* buf,
                                size_t len,
                                size_t* n_samples,
                                const char** err) {
    *n_samples = 0;
    if (len < 12 || memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0) {
        *err = "not a RIFF/WAVE file";
        return NULL;
    }
    size_t pos                = 12;
    int have_fmt              = 0;
    const unsigned char* data = NULL;
    size_t data_len           = 0;
    while (pos + 8 <= len) {
        uint32_t id_len   = rd32(buf + pos + 4);
        const unsigned char* body = buf + pos + 8;
        size_t avail      = len - (pos + 8);
        size_t chunk_len  = id_len <= avail ? id_len : avail;
        if (memcmp(buf + pos, "fmt ", 4) == 0) {
            if (chunk_len < 16) {
                *err = "fmt chunk too short";
                return NULL;
            }
            uint16_t format   = rd16(body);
            uint16_t channels = rd16(body + 2);
            uint32_t rate     = rd32(body + 4);
            uint16_t bits     = rd16(body + 14);
            if (format != 1 || channels != 1 || rate != 16000 || bits != 16) {
                *err = "audio must be PCM 16-bit, mono, 16000 Hz";
                return NULL;
            }
            have_fmt = 1;
        } else if (memcmp(buf + pos, "data", 4) == 0) {
            data     = body;
            data_len = chunk_len;
            break;
        }
        /* chunks are word aligned */
        size_t step = 8 + (size_t)id_len + (id_len & 1u);
        if (step > len - pos)
            break;
        pos += step;
    }
    if (!have_fmt) {
        *err = "missing fmt chunk";
        return NULL;
    }
    if (!data || data_len < 2) {
        *err = "missing or empty data chunk";
        return NULL;
    }
    size_t n   = data_len / 2;
    float* out = malloc(n * sizeof(float));
    if (!out) {
        *err = "out of memory";
        return NULL;
    }
    for (size_t i = 0; i < n; i++)
        out[i] = (float)(int16_t)rd16(data + 2 * i) / 32768.0f;
    *n_samples = n;
    return out;
}
