// GKDT ggml runtime - image loading, PIL-compatible geometry and preprocessing.
#include "image_io.hpp"
#include "common.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include <cmath>
#include <cstring>

#if defined(GKD_USE_OPENMP)
#include <omp.h>
#endif

namespace gkd {

bool load_image(const std::string& path, RgbImage& out) {
    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        GKD_LOG_ERROR("stb_image failed on %s: %s", path.c_str(), stbi_failure_reason());
        return false;
    }
    out.w = w;
    out.h = h;
    out.data.assign(data, data + (size_t)w * h * 3);
    stbi_image_free(data);
    return true;
}

// ---------------------------------------------------------------------------
// PIL-compatible antialiased bilinear resize (triangle filter).
// PIL samples with support = 1.0 * scale when downscaling, 1.0 when upscaling.
// ---------------------------------------------------------------------------
RgbImage resize_bilinear_pil(const RgbImage& src, int dst_w, int dst_h) {
    if (dst_w == src.w && dst_h == src.h) return src;

    const float scale_x = (float)dst_w / src.w;
    const float scale_y = (float)dst_h / src.h;

    // Pillow-exact precompute (Resample.c::precompute_coeffs, double precision):
    //   filterscale = max(1, src/dst); support = 1 * filterscale
    //   center = (j + 0.5) * src / dst
    //   taps x in [ceil(center - support), floor(center + support)] clamped,
    //   weight = tri((x + 0.5 - center) / filterscale) with tri: |x| < 1 -> 1-|x|,
    //   then normalized by the tap sum.
    struct AxisWeights {
        std::vector<int> i0, i1;
        std::vector<int> off;   // offset of this pixel's taps inside `w`
        std::vector<float> w;   // concatenated weights
    };
    auto build_axis = [&](int src_len, int dst_len) {
        double filterscale = (dst_len < src_len) ? (double)src_len / dst_len : 1.0;
        double support = 1.0 * filterscale;
        AxisWeights ax;
        ax.i0.resize(dst_len);
        ax.i1.resize(dst_len);
        ax.off.resize(dst_len);
        for (int j = 0; j < dst_len; j++) {
            double center = (j + 0.5) * (double)src_len / dst_len;
            int i0 = (int)std::ceil(center - support);
            int i1 = (int)std::floor(center + support);
            if (i0 < 0) i0 = 0;
            if (i1 > src_len - 1) i1 = src_len - 1;
            ax.i0[j] = i0;
            ax.i1[j] = i1;
            ax.off[j] = (int)ax.w.size();
            double sum = 0.0;
            for (int i = i0; i <= i1; i++) {
                double arg = (i - center + 0.5) / filterscale;
                double wt = (arg < -1.0 || arg >= 1.0) ? 0.0 : 1.0 - std::fabs(arg);
                ax.w.push_back((float)wt);
                sum += wt;
            }
            if (sum > 0.0) {
                for (int k = ax.off[j]; k < (int)ax.w.size(); k++) ax.w[k] = (float)(ax.w[k] / sum);
            }
        }
        return ax;
    };

    const AxisWeights xs = build_axis(src.w, dst_w);
    const AxisWeights ys = build_axis(src.h, dst_h);

    RgbImage dst(dst_w, dst_h);
#if defined(GKD_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float acc[3] = {0.f, 0.f, 0.f};
            const int y0 = ys.i0[y], y1 = ys.i1[y];
            const int x0 = xs.i0[x], x1 = xs.i1[x];
            const float* xw = &xs.w[xs.off[x]];
            for (int sy = y0; sy <= y1; sy++) {
                const uint8_t* srow = src.row(sy);
                float wy = ys.w[ys.off[y] + (sy - y0)];
                for (int sx = x0; sx <= x1; sx++) {
                    float w = wy * xw[sx - x0];
                    const uint8_t* px = srow + sx * 3;
                    acc[0] += w * px[0];
                    acc[1] += w * px[1];
                    acc[2] += w * px[2];
                }
            }
            uint8_t* d = dst.row(y) + x * 3;
            for (int c = 0; c < 3; c++) {
                int v = (int)std::lround(acc[c]);
                d[c] = (uint8_t)std::min(255, std::max(0, v));
            }
        }
    }
    return dst;
}

RgbImage pad_image(const RgbImage& src, int left, int top, int right, int bottom, uint8_t fill[3]) {
    RgbImage dst(src.w + left + right, src.h + top + bottom);
    for (int y = 0; y < dst.h; y++) {
        uint8_t* d = dst.row(y);
        for (int x = 0; x < dst.w; x++) {
            int sx = x - left, sy = y - top;
            if (sx >= 0 && sx < src.w && sy >= 0 && sy < src.h) {
                const uint8_t* s = src.row(sy) + sx * 3;
                d[x * 3 + 0] = s[0];
                d[x * 3 + 1] = s[1];
                d[x * 3 + 2] = s[2];
            } else {
                d[x * 3 + 0] = fill[0];
                d[x * 3 + 1] = fill[1];
                d[x * 3 + 2] = fill[2];
            }
        }
    }
    return dst;
}

RgbImage crop_image(const RgbImage& src, int x0, int y0, int x1, int y1) {
    x1 = std::min(x1, src.w);
    y1 = std::min(y1, src.h);
    int w = std::max(0, x1 - x0);
    int h = std::max(0, y1 - y0);
    RgbImage dst(w, h);
    for (int y = 0; y < h; y++) {
        std::memcpy(dst.row(y), src.row(y0 + y) + (size_t)x0 * 3, (size_t)w * 3);
    }
    return dst;
}

PreprocessResult preprocess_roi(const RgbImage& src, float bbox[4], int square,
                                const float* kps_in, int n_kps, const uint8_t* kps_vis_in) {
    const int w = src.w, h = src.h;
    // Input bbox is (x1, y1, x2, y2) top-left / bottom-right corners. The
    // official pipeline first converts it through bbox_check() into
    // (xmin, ymin, W, H), then RandomCrop(crop_gt_bbox=True) derives ltrb.
    float bx1 = bbox[0], by1 = bbox[1], bx2 = bbox[2], by2 = bbox[3];
    float xmin = std::min(std::max(bx1, 0.0f), (float)(w - 1));
    float ymin = std::min(std::max(by1, 0.0f), (float)(h - 1));
    float W = std::max(bx2 - bx1 + 1, 20.0f);
    float H = std::max(by2 - by1 + 1, 20.0f);
    W = std::min(W, (float)w - xmin);
    H = std::min(H, (float)h - ymin);
    if (W < 20 || H < 20) {
        xmin = 0; ymin = 0; W = (float)(w - 1); H = (float)(h - 1);
    }

    // RandomCrop(crop_gt_bbox=True): int() truncation + clamps
    int bxmin = std::max((int)xmin, 0);
    int bymin = std::max((int)ymin, 0);
    int bxmax = std::min((int)((float)xmin + W) + 1, w - 1);
    int bymax = std::min((int)((float)ymin + H) + 1, h - 1);
    if (bxmax <= bxmin || bymax <= bymin) {
        bxmin = 0; bymin = 0; bxmax = w - 1; bymax = h - 1;
    }
    PreprocessResult res;
    ScaleTrans& tr = res.trans;
    tr.scale = 1.0f;
    tr.offset_x = 0.0f;
    tr.offset_y = 0.0f;

    RgbImage cur = crop_image(src, bxmin, bymin, bxmax, bymax);  // PIL crop ltrb exclusive
    tr.offset_x += (float)bxmin;
    tr.offset_y += (float)bymin;

    // Resize(longer side = square, PIL bilinear)
    int cw = cur.w, ch = cur.h;
    float scale = (cw < ch) ? (float)square / ch : (float)square / cw;
    int tw = (cw < ch) ? (int)std::nearbyint(cw * scale) : square;
    int th = (cw < ch) ? square : (int)std::nearbyint(ch * scale);
    if (tw < 1) tw = 1;
    if (th < 1) th = 1;
    cur = resize_bilinear_pil(cur, tw, th);
    tr.scale *= scale;
    // offsets scale with the same factor (meta['offset'] *= scale)
    tr.offset_x *= scale;
    tr.offset_y *= scale;

    // CenterPad(square) with the ImageNet mean pixel
    int left = (int)((square - tw) / 2.0f);
    int top = (int)((square - th) / 2.0f);
    uint8_t fill[3] = {124, 116, 104};
    cur = pad_image(cur, left, top, square - tw - left, square - th - top, fill);
    tr.offset_x -= (float)left;   // meta['offset'] -= ltrb[:2]
    tr.offset_y -= (float)top;

    res.img = std::move(cur);

    // transform keypoints:  P' = P * scale - offset  (crop adds +ltrb via offset)
    if (kps_in && n_kps > 0) {
        res.kps_norm.resize((size_t)n_kps * 2);
        res.kps_valid.resize(n_kps);
        for (int k = 0; k < n_kps; k++) {
            float x = kps_in[k * 2 + 0];
            float y = kps_in[k * 2 + 1];
            uint8_t vis = kps_vis_in ? kps_vis_in[k] : 1;
            x = x * tr.scale - tr.offset_x;
            y = y * tr.scale - tr.offset_y;
            if (!vis) { x = 0; y = 0; }
            // CoordinateNormalize: (p / square - 0.5) * 2
            res.kps_norm[k * 2 + 0] = (x / square - 0.5f) * 2.0f;
            res.kps_norm[k * 2 + 1] = (y / square - 0.5f) * 2.0f;
            res.kps_valid[k] = vis;
        }
    }
    return res;
}

// out_chw / out_nhwc both use the ggml ne={W,H,C,B} convention, which in
// memory is the plain CHW (channel-major) image layout: index c*W*H + y*W + x.
static void normalize_pixels(const RgbImage& img, const float mean[3], const float stdv[3],
                             float* out_chw, float* out_nhwc) {
    const int w = img.w, h = img.h;
#if defined(GKD_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            const uint8_t* px = img.row(y);
            for (int x = 0; x < w; x++) {
                float v = (px[x * 3 + c] / 255.0f - mean[c]) / stdv[c];
                if (out_chw) out_chw[(size_t)c * w * h + y * w + x] = v;
                if (out_nhwc) out_nhwc[(size_t)c * w * h + y * w + x] = v;
            }
        }
    }
}

void image_to_chw_normalized(const RgbImage& img, const float mean[3], const float stdv[3],
                             float* out_chw) {
    normalize_pixels(img, mean, stdv, out_chw, nullptr);
}

void image_to_im2col_normalized(const RgbImage& img, const float mean[3], const float stdv[3],
                                int patch, float* out_im2col) {
    const int w = img.w, h = img.h;
    const int GW = w / patch;
    const int K = 3 * patch * patch;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            const uint8_t* px = img.row(y);
            const int hh = y / patch, kh = y % patch;
            for (int x = 0; x < w; x++) {
                float v = (px[x * 3 + c] / 255.0f - mean[c]) / stdv[c];
                const int ww = x / patch, kw = x % patch;
                // ggml {K, L} layout: token-major (element (k, l) at l*K + k)
                out_im2col[(size_t)(ww + hh * GW) * K + (c * patch * patch + kh * patch + kw)] = v;
            }
        }
    }
}

void recover_kps(const float* kps_norm, int n, int square_len, const ScaleTrans& trans, float* out) {
    for (int k = 0; k < n; k++) {
        float x = kps_norm[k * 2 + 0];
        float y = kps_norm[k * 2 + 1];
        x = x / 2.0f + 0.5f;
        y = y / 2.0f + 0.5f;
        x = x * square_len + trans.offset_x;
        y = y * square_len + trans.offset_y;
        out[k * 2 + 0] = x / trans.scale;
        out[k * 2 + 1] = y / trans.scale;
    }
}

}  // namespace gkd
