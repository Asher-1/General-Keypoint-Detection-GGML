// GKDT ggml runtime - output rendering (keypoint/skeleton visualization).
#pragma once

#include "image_io.hpp"

#include <string>
#include <vector>

namespace gkd {

struct KeypointStyle {
    float score_thresh = 0.05f;
    int   radius = 4;
    int   thickness = 2;
};

struct SkeletonLink {
    int a;  // 0-based keypoint index
    int b;
};

// Draw keypoints (scores below the threshold are skipped) and optional
// skeleton links, then write a JPEG. Keypoints are in ORIGINAL image coords.
bool render_and_save(const RgbImage& image, const float* kps_xy, const float* scores, int n,
                     const std::vector<SkeletonLink>& skeleton, const KeypointStyle& style,
                     const std::string& out_path);

}  // namespace gkd
