// GKDT ggml runtime - public C API.
//
// A minimal embedding surface over the three ggml graphs (DINOv3 vision tower,
// dinotxt text tower + adaptation net, KG transformer + detection head).
// One GkdSession holds the loaded GGUF weights and the cached graphs; a call
// to gkd_detect() reproduces the official PyTorch forward 1:1.
//
// Thread-safety: a session is single-threaded. Use one session per consumer,
// or serialize access externally. The CPU backend uses `n_threads` workers.
//
// Build: link against libgkdgml (which pulls in ggml and its enabled backends).
#ifndef GKDGML_H
#define GKDGML_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GKDGML_API extern

// Opaque session handle.
typedef struct gkd_session gkd_session_t;

// One detected keypoint: position in the ROI's normalized -1..1 space and the
// peak heatmap score in 0..~1.
typedef struct {
    float x, y;     // normalized -1..1 (input/ROI space)
    float score;    // heatmap peak value
} gkd_keypoint_t;

// Result of one detect() call over a single ROI.
typedef struct {
    int n_prompts;              // number of returned keypoints
    gkd_keypoint_t* keypoints;  // n_prompts entries (owned by the session)
} gkd_result_t;

// Create a session from a GGUF model produced by scripts/convert_gkd_to_gguf.py.
// n_threads <= 0 uses the hardware default. Returns NULL on failure.
GKDGML_API gkd_session_t* gkd_session_create(const char* gguf_path, int n_threads);

GKDGML_API void gkd_session_free(gkd_session_t* s);

// Human-readable backend name ("CPU", "CUDA0", "Vulkan0", ...).
GKDGML_API const char* gkd_session_backend(const gkd_session_t* s);

// General keypoint detection on one image.
//
//   image_bgr / image_rgb : w*h*3 pixel buffer, row-major
//   rgb_input             : 0 => image_bgr is BGR, 1 => RGB
//   bbox                  : optional ROI [x1,y1,x2,y2] in pixels (inclusive
//                           corners); NULL = whole image
//   kps_texts             : text prompts (may be NULL when visual prompts are
//                           given); n_texts entries, each NUL-terminated
//   support_rgb           : optional 1-shot support image (w*h*3, same layout
//                           flag as above); NULL for text-only mode
//   support_w/h           : support image dimensions
//   support_kps           : support keypoints [x1,y1, x2,y2, ...] in support
//                           image pixels; n_support_kps pairs, all valid
//   out                   : receives the keypoints (valid until the next call
//                           on the same session)
// Returns 0 on success, non-zero on failure.
GKDGML_API int gkd_detect(gkd_session_t* s,
                          const uint8_t* image_rgb, int w, int h,
                          const float* bbox,
                          const char* const* kps_texts, int n_texts,
                          const uint8_t* support_rgb, int support_w, int support_h,
                          const float* support_kps, int n_support_kps,
                          gkd_result_t* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GKDGML_H
