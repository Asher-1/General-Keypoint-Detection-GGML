// GKDT ggml runtime - output rendering (keypoint/skeleton visualization).
#include "postprocess.hpp"
#include "common.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace gkd {

namespace {
// palette matched to the official visualize_keypoints colors
const uint8_t kPalette[][3] = {
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255},
    {0, 255, 255}, {255, 128, 0}, {128, 0, 255}, {0, 255, 128}, {255, 0, 128},
    {128, 255, 0}, {0, 128, 255}, {255, 64, 64}, {64, 255, 64}, {64, 64, 255},
};

void put_pixel(RgbImage& img, int x, int y, const uint8_t c[3]) {
    if (x < 0 || y < 0 || x >= img.w || y >= img.h) return;
    uint8_t* p = img.row(y) + x * 3;
    p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
}

void draw_disk(RgbImage& img, float cx, float cy, float r, const uint8_t c[3]) {
    int x0 = (int)std::floor(cx - r), x1 = (int)std::ceil(cx + r);
    int y0 = (int)std::floor(cy - r), y1 = (int)std::ceil(cy + r);
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (d <= r) put_pixel(img, x, y, c);
        }
    }
}

void draw_line(RgbImage& img, float x0, float y0, float x1, float y1, int thickness, const uint8_t c[3]) {
    float dx = x1 - x0, dy = y1 - y0;
    int steps = (int)std::ceil(std::max(std::fabs(dx), std::fabs(dy)));
    steps = std::max(steps, 1);
    for (int i = 0; i <= steps; i++) {
        float t = (float)i / steps;
        float x = x0 + dx * t, y = y0 + dy * t;
        int r = thickness / 2;
        for (int oy = -r; oy <= r; oy++) {
            for (int ox = -r; ox <= r; ox++) {
                put_pixel(img, (int)std::lround(x) + ox, (int)std::lround(y) + oy, c);
            }
        }
    }
}
}  // namespace

bool render_and_save(const RgbImage& image, const float* kps_xy, const float* scores, int n,
                     const std::vector<SkeletonLink>& skeleton, const KeypointStyle& style,
                     const std::string& out_path) {
    RgbImage out = image;  // copy: we never touch the caller's tensor
    // skeleton first (under the keypoints)
    for (size_t s = 0; s < skeleton.size(); s++) {
        int a = skeleton[s].a, b = skeleton[s].b;
        if (a < 0 || b < 0 || a >= n || b >= n) continue;
        if (scores && (scores[a] < style.score_thresh || scores[b] < style.score_thresh)) continue;
        const uint8_t* c = kPalette[s % (sizeof(kPalette) / sizeof(kPalette[0]))];
        draw_line(out, kps_xy[a * 2], kps_xy[a * 2 + 1], kps_xy[b * 2], kps_xy[b * 2 + 1],
                  style.thickness + 1, c);
    }
    for (int j = 0; j < n; j++) {
        if (scores && scores[j] < style.score_thresh) continue;
        const uint8_t* c = kPalette[j % (sizeof(kPalette) / sizeof(kPalette[0]))];
        draw_disk(out, kps_xy[j * 2], kps_xy[j * 2 + 1], (float)style.radius, c);
    }
    if (!stbi_write_jpg(out_path.c_str(), out.w, out.h, 3, out.data.data(), 95)) {
        GKD_LOG_ERROR("failed to write %s", out_path.c_str());
        return false;
    }
    return true;
}

}  // namespace gkd
