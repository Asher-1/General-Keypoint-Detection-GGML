// GKDT ggml runtime - backend management.
// Persistent CPU threadpool + optional GPU backend spanned by a scheduler so
// unsupported ops fall back to CPU automatically (mirrors ultralytics-ggml).
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace gkd {

struct BackendCtx {
    int n_threads = 1;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t galloc = nullptr;

    bool has_gpu() const { return gpu != nullptr; }
};

// n_threads <= 0 keeps the hardware default. graph_nodes sizes the scheduler's
// node hash set (created once); a generous upper bound is fine.
BackendCtx init_backend_ctx(int n_threads, size_t graph_nodes);
void free_backend_ctx(BackendCtx& ctx);

ggml_backend_buffer_type_t backend_weight_buft(const BackendCtx& ctx);
// CPU path: pass a per-graph gallocr (graphs must not share one gallocr,
// a resize triggered by another graph invalidates earlier allocations).
bool backend_graph_alloc(BackendCtx& ctx, ggml_cgraph* graph, ggml_gallocr_t& galloc);
int  backend_graph_compute(BackendCtx& ctx, ggml_cgraph* graph);
const char* backend_name(const BackendCtx& ctx);
const char* backend_device_desc();

}  // namespace gkd
