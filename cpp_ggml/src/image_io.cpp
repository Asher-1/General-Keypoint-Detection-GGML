// GKDT ggml runtime - image loading, PIL-compatible geometry and preprocessing.
#include "image_io.hpp"
#include "common.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(GKD_HAVE_LIBJPEG)
#include <csetjmp>
#include <cstdio>
#include <jpeglib.h>
#endif

#if defined(GKD_USE_OPENMP)
#include <omp.h>
#endif

namespace gkd {

#if defined(GKD_HAVE_LIBJPEG)
namespace {

struct JpegErrorMgr {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

void jpeg_error_exit(j_common_ptr cinfo) {
    longjmp(reinterpret_cast<JpegErrorMgr*>(cinfo->err)->setjmp_buffer, 1);
}

// libjpeg decode with library defaults - the exact decoder behind cv2.imread
// and PIL (islow IDCT, fancy upsampling), so JPEG inputs are bit-identical to
// the official pipeline. (stb's own decoder deviates by up to +-3/255.)
bool load_image_jpeg_libjpeg(const std::string& path, RgbImage& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    jpeg_decompress_struct cinfo{};
    JpegErrorMgr jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(f);
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(f);
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    if (cinfo.output_components != 3) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(f);
        return false;
    }
    RgbImage img(cinfo.output_width, cinfo.output_height);
    while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t* row = img.row(cinfo.output_scanline);
        JSAMPROW rows[1] = {row};
        jpeg_read_scanlines(&cinfo, rows, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(f);
    out = std::move(img);
    return true;
}

}  // namespace
#endif  // GKD_HAVE_LIBJPEG

bool load_image(const std::string& path, RgbImage& out) {
#if defined(GKD_HAVE_LIBJPEG)
    // JPEG magic (FF D8): decode through libjpeg for official-pipeline parity
    {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f) {
            unsigned char magic[2] = {0, 0};
            size_t n = std::fread(magic, 1, 2, f);
            std::fclose(f);
            if (n == 2 && magic[0] == 0xFF && magic[1] == 0xD8 &&
                load_image_jpeg_libjpeg(path, out)) {
                return true;
            }
        }
    }
#endif
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
// Pillow bit-exact antialiased bilinear resize (triangle filter).
//
// Port of Pillow src/libImaging/Resample.c (verified against Pillow 12.0.0),
// which is exactly what the official pipeline's Image.resize(BILINEAR) runs:
//   precompute per axis (double):
//     filterscale = scale = src/dst;  if (filterscale < 1) filterscale = 1
//     support = 1.0 * filterscale;  ksize = ceil(support)*2 + 1
//     xmin = (int)(center - support + 0.5) clamped to 0   // C trunc!
//     xmax = (int)(center + support + 0.5) clamped to src // tap COUNT
//     w[x] = tri((x + xmin - center + 0.5) / filterscale), normalized by sum
//   fixed point: PRECISION_BITS = 22; k[x] = (int)(0.5 + w[x] * (1 << 22))
//   pass (INT32 accumulation, uint8 intermediate image):
//     acc = 1 << 21;  acc += pixel * k[x];  out = clamp(acc >> 22, 0, 255)
//   order: HORIZONTAL pass into a uint8 buffer, then VERTICAL pass.
//
// The per-pass uint8 rounding is part of the reference behavior: collapsing
// both passes into a single float accumulation (our previous implementation)
// shifts values by ~1 LSB, which propagated through the 24-block network as
// preprocessing noise and caused near-tie heatmap argmax flips.
// ---------------------------------------------------------------------------
RgbImage resize_bilinear_pil(const RgbImage& src, int dst_w, int dst_h) {
    if (dst_w == src.w && dst_h == src.h) return src;

    constexpr int PRECISION_BITS = 22;  // 32 - 8 - 2 in Resample.c
    auto clip8 = [](int32_t v) {
        // arithmetic shift == floor; the lookup table in Resample.c clamps
        return (uint8_t)std::min(255, std::max(0, (int)(v >> PRECISION_BITS)));
    };

    // per-axis coefficient table, exactly Resample.c::precompute_coeffs
    struct AxisCoefs {
        int ksize = 0;
        std::vector<int> xmin, cnt;
        std::vector<int32_t> k;  // dst_len * ksize, row-major
    };
    auto build_axis = [&](int src_len, int dst_len) {
        AxisCoefs ax;
        double scale = (double)src_len / dst_len;
        double filterscale = scale < 1.0 ? 1.0 : scale;
        double support = 1.0 * filterscale;
        ax.ksize = (int)std::ceil(support) * 2 + 1;
        ax.xmin.resize(dst_len);
        ax.cnt.resize(dst_len);
        ax.k.assign((size_t)dst_len * ax.ksize, 0);
        for (int xx = 0; xx < dst_len; xx++) {
            double center = (xx + 0.5) * scale;  // in0 = 0 (full-image box)
            double ss = 1.0 / filterscale;
            int xmin = (int)(center - support + 0.5);  // C trunc, like the reference
            if (xmin < 0) xmin = 0;
            int xmax = (int)(center + support + 0.5);
            if (xmax > src_len) xmax = src_len;
            int n = xmax - xmin;
            ax.xmin[xx] = xmin;
            ax.cnt[xx] = n;
            double ww = 0.0;
            std::vector<double> w(n);
            for (int x = 0; x < n; x++) {
                double arg = (x + xmin - center + 0.5) * ss;
                if (arg < 0) arg = -arg;
                w[x] = arg < 1.0 ? 1.0 - arg : 0.0;
                ww += w[x];
            }
            for (int x = 0; x < n; x++) {
                if (ww != 0.0) w[x] /= ww;
                // normalize_coeffs_8bpc: (int)(0.5 + w * (1 << PRECISION_BITS))
                ax.k[(size_t)xx * ax.ksize + x] =
                    (int32_t)(0.5 + w[x] * (float)(1 << PRECISION_BITS));
            }
        }
        return ax;
    };

    const AxisCoefs xs = build_axis(src.w, dst_w);
    const AxisCoefs ys = build_axis(src.h, dst_h);

    // horizontal pass into a uint8 intermediate (pass order matters)
    RgbImage tmp(dst_w, src.h);
#if defined(GKD_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < src.h; y++) {
        const uint8_t* srow = src.row(y);
        uint8_t* drow = tmp.row(y);
        for (int x = 0; x < dst_w; x++) {
            const int xmin = xs.xmin[x], n = xs.cnt[x];
            const int32_t* k = &xs.k[(size_t)x * xs.ksize];
            int32_t a0 = 1 << (PRECISION_BITS - 1);
            int32_t a1 = a0, a2 = a0;
            for (int i = 0; i < n; i++) {
                const uint8_t* px = srow + (xmin + i) * 3;
                a0 += px[0] * k[i];
                a1 += px[1] * k[i];
                a2 += px[2] * k[i];
            }
            drow[x * 3 + 0] = clip8(a0);
            drow[x * 3 + 1] = clip8(a1);
            drow[x * 3 + 2] = clip8(a2);
        }
    }

    RgbImage dst(dst_w, dst_h);
#if defined(GKD_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst_h; y++) {
        const int ymin = ys.xmin[y], n = ys.cnt[y];
        const int32_t* k = &ys.k[(size_t)y * ys.ksize];
        uint8_t* drow = dst.row(y);
        for (int x = 0; x < dst_w; x++) {
            int32_t a0 = 1 << (PRECISION_BITS - 1);
            int32_t a1 = a0, a2 = a0;
            for (int i = 0; i < n; i++) {
                const uint8_t* px = tmp.row(ymin + i) + x * 3;
                a0 += px[0] * k[i];
                a1 += px[1] * k[i];
                a2 += px[2] * k[i];
            }
            drow[x * 3 + 0] = clip8(a0);
            drow[x * 3 + 1] = clip8(a1);
            drow[x * 3 + 2] = clip8(a2);
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
