// GKDT ggml runtime - C API implementation over the internal C++ session.
#include "gkdgml.h"

#include "gkd_graph.hpp"
#include "image_io.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace gkd;

struct gkd_session {
    std::unique_ptr<GkdSession> impl;
    // keep the last result alive so the returned pointers stay valid
    std::vector<gkd_keypoint_t> last;
};

gkd_session_t* gkd_session_create(const char* gguf_path, int n_threads) {
    auto s = new gkd_session();
    s->impl = GkdSession::create(gguf_path ? gguf_path : "", n_threads);
    if (!s->impl) {
        delete s;
        return nullptr;
    }
    return s;
}

void gkd_session_free(gkd_session_t* s) { delete s; }

const char* gkd_session_backend(const gkd_session_t* s) {
    return s && s->impl ? s->impl->backend() : "";
}

int gkd_detect(gkd_session_t* s,
               const uint8_t* image_rgb, int w, int h,
               const float* bbox,
               const char* const* kps_texts, int n_texts,
               const uint8_t* support_rgb, int support_w, int support_h,
               const float* support_kps, int n_support_kps,
               gkd_result_t* out) {
    if (!s || !s->impl || !image_rgb || w <= 0 || h <= 0 || !out) return -1;

    // wrap the caller's buffer without copying (the runtime only reads it)
    RgbImage query;
    query.w = w;
    query.h = h;
    query.data.assign(image_rgb, image_rgb + (size_t)w * h * 3);

    DetectInput extra;
    for (int i = 0; i < n_texts; i++) {
        if (kps_texts[i]) extra.kps_texts.push_back(kps_texts[i]);
    }
    if (support_rgb && support_w > 0 && support_h > 0 && support_kps && n_support_kps > 0) {
        RgbImage support;
        support.w = support_w;
        support.h = support_h;
        support.data.assign(support_rgb, support_rgb + (size_t)support_w * support_h * 3);
        extra.support_image = std::move(support);
        extra.has_support = true;
        extra.support_kps_xy.assign(support_kps, support_kps + (size_t)n_support_kps * 2);
        extra.support_kps_vis.assign(n_support_kps, 1);
    }

    std::vector<DetectOutput> results;
    if (!s->impl->detect(query, bbox, bbox ? 1 : 0, extra, results)) return -2;
    if (results.empty()) return -3;

    const DetectOutput& r = results[0];
    s->last.clear();
    s->last.reserve(r.n_prompts);
    for (int i = 0; i < r.n_prompts; i++) {
        s->last.push_back({r.kps_norm[(size_t)i * 2 + 0],
                           r.kps_norm[(size_t)i * 2 + 1],
                           r.scores[i]});
    }
    out->n_prompts = (int)s->last.size();
    out->keypoints = s->last.data();
    return 0;
}
