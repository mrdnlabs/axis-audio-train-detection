#include "model.h"

#include <errno.h>
#include <fcntl.h>
#include <glib.h>
#include <larod.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

struct model {
    larodConnection* conn;
    larodModel* model;
    larodTensor** inputs;
    larodTensor** outputs;
    size_t n_inputs;
    size_t n_outputs;
    larodJobRequest* req;
    int model_fd;
    void* in_map;
    size_t in_size;
    void* out_map[2];
    size_t out_size[2];
    int prob_idx;
    int scores_idx;
    char device[64];
    GMutex lock;
};

static void set_err(char* err, size_t len, const char* what, larodError* e) {
    snprintf(err, len, "%s: %s", what, e && e->msg ? e->msg : "unknown error");
}

static void* map_tensor(larodTensor* t, size_t* size, larodError** e) {
    int fd = larodGetTensorFd(t, e);
    if (fd == LAROD_INVALID_FD)
        return NULL;
    if (!larodGetTensorFdSize(t, size, e))
        return NULL;
    void* p = mmap(NULL, *size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return p == MAP_FAILED ? NULL : p;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

model_t* model_new(const char* path, const char* devices, char* err, size_t err_len) {
    larodError* e = NULL;
    model_t* m    = calloc(1, sizeof(*m));
    if (!m) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    g_mutex_init(&m->lock);
    m->model_fd   = -1;
    m->prob_idx   = -1;
    m->scores_idx = -1;

    if (!larodConnect(&m->conn, &e)) {
        set_err(err, err_len, "larodConnect", e);
        goto fail;
    }

    size_t n_dev                  = 0;
    const larodDevice** dev_list  = larodListDevices(m->conn, &n_dev, &e);
    GString* names                = g_string_new(NULL);
    for (size_t i = 0; dev_list && i < n_dev; i++) {
        const char* name = larodGetDeviceName(dev_list[i], NULL);
        g_string_append_printf(names, "%s%s", i ? "," : "", name ? name : "?");
    }
    syslog(LOG_INFO, "[traindet] larod devices: %s", names->str);
    g_string_free(names, TRUE);
    larodClearError(&e); /* dev_list is owned by the connection: must not be freed */

    m->model_fd = open(path, O_RDONLY);
    if (m->model_fd < 0) {
        snprintf(err, err_len, "open %s: %s", path, strerror(errno));
        goto fail;
    }

    gchar** wanted = g_strsplit(devices, ",", -1);
    for (int i = 0; wanted[i] && !m->model; i++) {
        const larodDevice* dev = larodGetDevice(m->conn, g_strstrip(wanted[i]), 0, &e);
        if (!dev) {
            larodClearError(&e);
            continue;
        }
        m->model = larodLoadModel(m->conn, m->model_fd, dev, LAROD_ACCESS_PRIVATE, "traindet", NULL, &e);
        if (m->model) {
            snprintf(m->device, sizeof(m->device), "%s", wanted[i]);
        } else {
            syslog(LOG_WARNING, "[traindet] load on %s failed: %s", wanted[i], e ? e->msg : "?");
            larodClearError(&e);
        }
    }
    g_strfreev(wanted);
    if (!m->model) {
        snprintf(err, err_len, "could not load model on any of: %s", devices);
        goto fail;
    }

    m->inputs = larodAllocModelInputs(m->conn, m->model, LAROD_FD_PROP_MAP | LAROD_FD_PROP_READWRITE,
                                      &m->n_inputs, NULL, &e);
    if (!m->inputs || m->n_inputs != 1) {
        set_err(err, err_len, "alloc inputs", e);
        goto fail;
    }
    m->outputs = larodAllocModelOutputs(m->conn, m->model, LAROD_FD_PROP_MAP | LAROD_FD_PROP_READWRITE,
                                        &m->n_outputs, NULL, &e);
    if (!m->outputs || m->n_outputs != 2) {
        set_err(err, err_len, "alloc outputs (expected 2)", e);
        goto fail;
    }
    m->in_map = map_tensor(m->inputs[0], &m->in_size, &e);
    if (!m->in_map || m->in_size < MODEL_INPUT_SAMPLES * sizeof(float)) {
        snprintf(err, err_len, "input tensor map failed or too small (%zu bytes)", m->in_size);
        goto fail;
    }
    for (int i = 0; i < 2; i++) {
        m->out_map[i] = map_tensor(m->outputs[i], &m->out_size[i], &e);
        if (!m->out_map[i]) {
            set_err(err, err_len, "output tensor map", e);
            goto fail;
        }
        /* Identify outputs by size: 1 float = train prob, 521 floats = YAMNet scores. */
        size_t bytes = 0;
        larodGetTensorByteSize(m->outputs[i], &bytes, NULL);
        if (bytes == 0)
            bytes = m->out_size[i];
        if (bytes == sizeof(float))
            m->prob_idx = i;
        else if (bytes == MODEL_NUM_CLASSES * sizeof(float))
            m->scores_idx = i;
    }
    if (m->prob_idx < 0 || m->scores_idx < 0) {
        snprintf(err, err_len, "unexpected output tensor sizes (%zu, %zu)", m->out_size[0], m->out_size[1]);
        goto fail;
    }
    m->req = larodCreateJobRequest(m->model, m->inputs, m->n_inputs, m->outputs, m->n_outputs, NULL, &e);
    if (!m->req) {
        set_err(err, err_len, "create job request", e);
        goto fail;
    }
    syslog(LOG_INFO, "[traindet] model %s loaded on %s", path, m->device);
    return m;

fail:
    larodClearError(&e);
    model_free(m);
    return NULL;
}

bool model_run(model_t* m, const float* samples, model_output_t* out, char* err, size_t err_len) {
    larodError* e = NULL;
    bool ok       = true;
    g_mutex_lock(&m->lock);
    double t0 = now_ms();
    memcpy(m->in_map, samples, MODEL_INPUT_SAMPLES * sizeof(float));
    if (!larodRunJob(m->conn, m->req, &e)) {
        set_err(err, err_len, "larodRunJob", e);
        larodClearError(&e);
        ok = false;
    } else {
        out->train_prob = ((const float*)m->out_map[m->prob_idx])[0];
        memcpy(out->scores, m->out_map[m->scores_idx], sizeof(out->scores));
        out->latency_ms = now_ms() - t0;
    }
    g_mutex_unlock(&m->lock);
    return ok;
}

const char* model_device(const model_t* m) {
    return m->device;
}

void model_free(model_t* m) {
    if (!m)
        return;
    if (m->in_map)
        munmap(m->in_map, m->in_size);
    for (int i = 0; i < 2; i++)
        if (m->out_map[i])
            munmap(m->out_map[i], m->out_size[i]);
    larodDestroyJobRequest(&m->req);
    if (m->inputs)
        larodDestroyTensors(m->conn, &m->inputs, m->n_inputs, NULL);
    if (m->outputs)
        larodDestroyTensors(m->conn, &m->outputs, m->n_outputs, NULL);
    larodDestroyModel(&m->model);
    if (m->conn)
        larodDisconnect(&m->conn, NULL);
    if (m->model_fd >= 0)
        close(m->model_fd);
    g_mutex_clear(&m->lock);
    free(m);
}
