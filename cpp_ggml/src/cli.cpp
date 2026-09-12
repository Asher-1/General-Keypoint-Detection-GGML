// GKDT ggml runtime - gkd-cli: detect / bench / info subcommands.
//
//   gkd-cli detect --model <gguf> --input <img.jpg> [--kps-texts ...]
//                  [--support-image ... --support-kps ...] [--bbox x1 y1 x2 y2]
//                  [--out result.jpg] [--threads N]
//   gkd-cli bench  --model <gguf> --input <img.jpg> [--warmup 10 --iters 50]
//   gkd-cli info   --model <gguf>
#include "gkd_graph.hpp"
#include "postprocess.hpp"
#include "common.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace gkd;

namespace {

void usage() {
    std::fprintf(stderr,
        "usage: gkd-cli <command> [options]\n"
        "\n"
        "commands:\n"
        "  detect  general keypoint detection on one image\n"
        "  bench   steady-state latency benchmark\n"
        "  info    print model metadata\n"
        "\n"
        "detect options:\n"
        "  --model PATH            GGUF model (required)\n"
        "  --input PATH ...        query image(s); bench/detect accept several\n"
        "  --bbox X1 Y1 X2 Y2      ROI box; whole image when omitted\n"
        "  --kps-texts ...         keypoint name prompts (nargs)\n"
        "  --support-image PATH    1-shot support image\n"
        "  --support-kps X Y ...   support keypoints in the support image\n"
        "  --skeleton A B ...      1-based skeleton links for rendering\n"
        "  --out PATH              rendered JPEG output\n"
        "  --threads N             CPU threads (default: hardware)\n"
        "  --dump-taps DIR         dump internal tensors for parity checks\n"
        "bench options: --model --input --bbox --kps-texts --support-image\n"
        "               --support-kps --warmup --iters --threads\n");
}

struct Args {
    std::string cmd;
    std::string model, out, dump_taps;
    std::vector<std::string> inputs;   // one or more query images
    std::vector<float> bbox;
    std::vector<std::string> kps_texts;
    std::string support_image;
    std::vector<float> support_kps;
    std::vector<int> skeleton;
    int threads = -1;
    int warmup = 10, iters = 50;
};

// A repeatable numeric option (--bbox / --support-kps) consumes values until
// the next option. Detection boxes can legitimately be NEGATIVE (an object
// touching the image border), so "starts with '-'" is not a valid terminator:
// test whether the token parses as a float instead.
static bool looks_like_float(const char* s) {
    if (!s || !*s) return false;
    char* end = nullptr;
    std::strtof(s, &end);
    return end && *end == '\0';
}

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.cmd = argv[1];
    if (a.cmd != "detect" && a.cmd != "bench" && a.cmd != "info") return false;
    for (int i = 2; i < argc; i++) {
        std::string s = argv[i];
        auto next_str = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        auto next_f = [&]() -> float { return (i + 1 < argc) ? (float)atof(argv[++i]) : 0.0f; };
        auto next_i = [&]() -> int { return (i + 1 < argc) ? atoi(argv[++i]) : 0; };
        if (s == "--model") a.model = next_str();
        else if (s == "--input") { while (i + 1 < argc && argv[i + 1][0] != '-') a.inputs.push_back(argv[++i]); }
        else if (s == "--out") a.out = next_str();
        else if (s == "--dump-taps") a.dump_taps = next_str();
        else if (s == "--bbox") { while (i + 1 < argc && looks_like_float(argv[i + 1])) a.bbox.push_back(next_f()); }
        else if (s == "--kps-texts" || s == "--kps-textS") { while (i + 1 < argc && argv[i + 1][0] != '-') a.kps_texts.push_back(argv[++i]); }
        else if (s == "--support-image") a.support_image = next_str();
        else if (s == "--support-kps") { while (i + 1 < argc && looks_like_float(argv[i + 1])) a.support_kps.push_back(next_f()); }
        else if (s == "--skeleton") { while (i + 1 < argc && argv[i + 1][0] != '-') a.skeleton.push_back(next_i()); }
        else if (s == "--threads") a.threads = next_i();
        else if (s == "--warmup") a.warmup = next_i();
        else if (s == "--iters") a.iters = next_i();
        else { std::fprintf(stderr, "unknown option: %s\n", s.c_str()); return false; }
    }
    // info is metadata-only: model is enough; detect/bench need an input image
    if (a.cmd == "info") return !a.model.empty();
    return !a.model.empty() && !a.inputs.empty();
}

int cmd_info(const Args& a) {
    auto s = GkdSession::create(a.model, 1);
    if (!s) return 1;
    const ModelParams& P = s->params();
    std::printf("backend:          %s (%s build)\n", s->backend(), backend_device_desc());
    std::printf("architecture:     GKDT = DINOv3 ViT + dinotxt + KGTransformer + DetectionHead\n");
    std::printf("visual encoder:   D=%d depth=%d heads=%d patch=%d storage=%d img=%d feat=%dx%d\n",
                P.D, P.blocks, P.heads, P.patch, P.n_storage, P.img_size, P.feat_w, P.feat_w);
    std::printf("text encoder:     D=%d depth=%d heads=%d ctx=%d vocab=%d proj=%d\n",
                P.T_D, P.T_layers, P.T_heads, P.T_ctx, P.T_vocab, P.T_proj_dim);
    std::printf("adaptation net:   in=%d model=%d blocks=%d out=%d (half=%d)\n",
                P.A_in, P.A_model, P.A_blocks, P.A_out, P.text_half);
    std::printf("kernel generator: D=%d blocks=%d heads=%d d_ff=%d mask_token=%d\n",
                P.K_D, P.K_blocks, P.K_heads, P.K_ff, P.K_use_mask_token);
    std::printf("detection head:   bilinear x%d, kernel_norm=%d, heatmap=%dx%d\n",
                P.up_scale, P.kernel_norm, P.heat_w(), P.heat_w());
    std::printf("prompting:        sigma=%.1f pad_kps=%d\n", P.sigma, P.pad_kps);
    return 0;
}

int cmd_detect_or_bench(const Args& a) {
    RgbImage image;
    if (!load_image(a.inputs[0], image)) return 1;

    DetectInput extra;
    if (!a.support_image.empty()) {
        if (!load_image(a.support_image, extra.support_image)) return 1;
        extra.has_support = true;
        extra.support_kps_xy = a.support_kps;
        extra.support_kps_vis.assign(a.support_kps.size() / 2, 1);
    }
    extra.kps_texts = a.kps_texts;

    auto session = GkdSession::create(a.model, a.threads);
    if (!session) return 1;
    if (!a.dump_taps.empty()) session->set_dump_dir(a.dump_taps);

    const float* bbox = a.bbox.size() >= 4 ? a.bbox.data() : nullptr;

    if (a.cmd == "detect") {
        std::vector<DetectOutput> out;
        StageTiming t;
        int n_roi = (int)a.bbox.size() / 4;
        if (!session->detect(image, bbox, n_roi, extra, out, &t)) return 1;

        std::printf("backend: %s | rois: %d | prompts: %d | total: %.1f ms (preproc %.1f, vision %.1f, text %.1f, prompts %.1f, detect %.1f, decode %.1f)\n",
                    session->backend(), n_roi > 0 ? n_roi : 1, out[0].n_prompts, t.total, t.preprocess,
                    t.vision, t.text, t.prompt_prep, t.detect, t.decode);

        // one machine-readable line per ROI, each with its own ScaleTrans
        // (official single_obj_gkd_inference returns N_bbox x N x 2 predictions)
        for (int roi = 0; roi < (n_roi > 0 ? n_roi : 1); roi++) {
            const DetectOutput& r = out[roi];
            for (int j = 0; j < r.n_prompts; j++) {
                std::printf("  roi %d kp %2d: norm=(%+.4f, %+.4f) score=%.4f\n", roi, j,
                            r.kps_norm[j * 2], r.kps_norm[j * 2 + 1], r.scores[j]);
            }
            // rebuild this ROI's ScaleTrans (identical math to preprocess_roi)
            float bb[4] = {0, 0, (float)(image.w - 1), (float)(image.h - 1)};
            if (a.bbox.size() >= 4 * (roi + 1))
                std::memcpy(bb, a.bbox.data() + 4 * roi, sizeof(float) * 4);
            PreprocessResult pr = preprocess_roi(image, bb, session->params().img_size, nullptr, 0, nullptr);

            std::printf("{\"json\":{\"backend\":\"%s\",\"input\":\"%s\",\"roi\":%d,\"n_prompts\":%d,"
                        "\"trans\":[%.8f,%.8f,%.8f],\"kps_norm\":[",
                        session->backend(), a.inputs[0].c_str(), roi, r.n_prompts,
                        pr.trans.scale, pr.trans.offset_x, pr.trans.offset_y);
            for (int j = 0; j < r.n_prompts; j++)
                std::printf("%s%.6f,%.6f", j ? "," : "", r.kps_norm[j * 2], r.kps_norm[j * 2 + 1]);
            std::printf("],\"scores\":[");
            for (int j = 0; j < r.n_prompts; j++)
                std::printf("%s%.6f", j ? "," : "", r.scores[j]);
            std::printf("]}}\n");

            if (!a.out.empty()) {
                std::vector<float> kps_orig((size_t)r.n_prompts * 2);
                recover_kps(r.kps_norm.data(), r.n_prompts, session->params().img_size,
                            pr.trans, kps_orig.data());
                std::vector<SkeletonLink> links;
                for (size_t i = 0; i + 1 < a.skeleton.size(); i += 2) {
                    links.push_back({a.skeleton[i] - 1, a.skeleton[i + 1] - 1});
                }
                KeypointStyle style;
                std::string out_path = a.out;
                if (n_roi > 1) {
                    std::string base = a.out.substr(0, a.out.rfind('.'));
                    std::string ext = a.out.substr(a.out.rfind('.'));
                    out_path = base + "_roi" + std::to_string(roi) + ext;
                }
                render_and_save(image, kps_orig.data(), r.scores.data(), r.n_prompts, links, style, out_path);
                std::printf("rendered: %s\n", out_path.c_str());
            }
        }
        return 0;
    }

    // bench: every --input is benched within this single session
    for (const std::string& path : a.inputs) {
        RgbImage img;
        if (!load_image(path, img)) return 1;
        GkdSession::BenchWorkload wl;
        wl.query_image = img;
        if (a.bbox.size() >= 4) std::memcpy(wl.bbox, a.bbox.data(), sizeof(float) * 4);
        else { wl.bbox[2] = (float)(img.w - 1); wl.bbox[3] = (float)(img.h - 1); }
        wl.extra = extra;
        StageTiming avg;
        if (!session->bench(wl, a.warmup, a.iters, avg)) return 1;

        std::printf("{\"backend\":\"%s\",\"input\":\"%s\",\"n_texts\":%d,\"has_support\":%s,"
                    "\"warmup\":%d,\"iters\":%d,"
                    "\"preprocess_ms\":%.3f,\"vision_ms\":%.3f,\"text_ms\":%.3f,"
                    "\"prompt_prep_ms\":%.3f,\"detect_ms\":%.3f,\"decode_ms\":%.3f,"
                    "\"total_ms\":%.3f}\n",
                    session->backend(), path.c_str(), (int)a.kps_texts.size(),
                    extra.has_support ? "true" : "false", a.warmup, a.iters,
                    avg.preprocess, avg.vision, avg.text, avg.prompt_prep, avg.detect,
                    avg.decode, avg.total);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        usage();
        return 1;
    }
    if (a.cmd == "info") return cmd_info(a);
    return cmd_detect_or_bench(a);
}
