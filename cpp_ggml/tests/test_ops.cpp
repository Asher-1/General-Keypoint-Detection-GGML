// Golden-op tests for the graph pieces the GKDT runtime relies on.
// Each test builds a small graph, fills inputs, computes on the CPU backend
// and compares against an independent naive computation:
//   - mul_mat        {K,M}x{K,N} -> {M,N}   (weight layout: raw torch [out,in])
//   - flash_attn_ext {dh,S,H,B} strided head views of a packed qkv
//   - rope           out = x*cos + rot_half(x)*sin  (split formulation)
//   - l2 normalize   x / max(||x||, eps) along ne0
//   - interpolate    bilinear x4, align_corners=False semantics
// Build with -DGKD_GGML_BUILD_TESTS=ON; run gkd-test-ops, exit 0 = all pass.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_failures = 0;

static void check(const char* name, const std::vector<float>& got,
                  const std::vector<double>& want, double tol) {
    double maxerr = 0;
    for (size_t i = 0; i < want.size() && i < got.size(); i++)
        maxerr = std::max(maxerr, std::fabs((double)got[i] - want[i]));
    bool ok = maxerr <= tol;
    if (!ok) g_failures++;
    printf("%-28s %s (maxerr=%.3e tol=%.0e)\n", name, ok ? "PASS" : "FAIL", maxerr, tol);
}

static ggml_backend_t g_backend = nullptr;

struct TestCtx {
    ggml_backend_t backend = g_backend;
    ggml_context* ctx = nullptr;      // tensor metadata (+ graph), no_alloc
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph* gf = nullptr;
    std::vector<ggml_tensor*> inputs;  // allocated by the gallocr, filled by set()

    ~TestCtx() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
    }

    ggml_tensor* tensor_2d(int ne0, int ne1) {
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        inputs.push_back(t);
        return t;
    }

    void alloc_graph(ggml_tensor* out, int n_threads) {
        gf = ggml_new_graph_custom(ctx, 512, false);
        ggml_build_forward_expand(gf, out);
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(galloc, gf);
        // fill the input leaves now that the gallocr gave them buffers
        for (ggml_tensor* t : inputs) {
            std::mt19937 rng((uintptr_t)t * 2654435761u);
            std::uniform_real_distribution<float> d(-1, 1);
            std::vector<float> host(ggml_nelements(t));
            for (auto& v : host) v = d(rng);
            ggml_backend_tensor_set(t, host.data(), 0, host.size() * 4);
        }
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    void run() {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(2);
        }
    }

    std::vector<float> get(const ggml_tensor* t) const {
        std::vector<float> r(ggml_nelements(t));
        ggml_backend_tensor_get(t, r.data(), 0, r.size() * sizeof(float));
        return r;
    }
};

static std::mt19937 g_rng(7);
static std::uniform_real_distribution<float> g_dist(-1, 1);
static std::vector<float> randoms(size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = g_dist(g_rng);
    return v;
}

// ---- 1. mul_mat with the runtime's weight layout ----
static void test_mul_mat() {
    const int K = 768, M = 64, N = 32;
    std::vector<float> W = randoms((size_t)K * M), X = randoms((size_t)K * N);

    TestCtx t;
    t.ctx = ggml_init({1 << 26, nullptr, true});
    ggml_tensor* w = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor* x = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, K, N);
    ggml_tensor* y = ggml_mul_mat(t.ctx, w, x);
    t.inputs = {w, x};
    t.alloc_graph(y, 4);
    ggml_backend_tensor_set(w, W.data(), 0, W.size() * 4);
    ggml_backend_tensor_set(x, X.data(), 0, X.size() * 4);
    t.run();
    auto r = t.get(y);

    std::vector<double> want((size_t)M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)W[k + (size_t)m * K] * X[k + (size_t)n * K];
            want[(size_t)m + (size_t)n * M] = acc;
        }
    check("mul_mat layout", r, want, 1e-3);
}

// ---- 2. flash attention over packed-qkv strided head views ----
static void test_flash_attn() {
    const int D = 512, H = 8, dh = D / H, S = 33, L = 97;
    std::vector<float> Q = randoms((size_t)D * S), K = randoms((size_t)D * L), V = randoms((size_t)D * L);

    TestCtx t;
    t.ctx = ggml_init({1 << 26, nullptr, true});
    ggml_tensor* q = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, D, S);
    ggml_tensor* k = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, D, L);
    ggml_tensor* v = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, D, L);
    const size_t es = 4;
    ggml_tensor* qv = ggml_view_4d(t.ctx, q, dh, S, H, 1, q->nb[1], dh * es, dh * es * H, 0);
    ggml_tensor* kv = ggml_view_4d(t.ctx, k, dh, L, H, 1, k->nb[1], dh * es, dh * es * H, 0);
    ggml_tensor* vv = ggml_view_4d(t.ctx, v, dh, L, H, 1, v->nb[1], dh * es, dh * es * H, 0);
    ggml_tensor* attn = ggml_flash_attn_ext(t.ctx, qv, kv, vv, nullptr, 1.f / std::sqrt((float)dh), 0, 0);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor* packed = ggml_reshape_2d(t.ctx, attn, D, S);  // head-major packing
    t.inputs = {q, k, v};
    t.alloc_graph(packed, 4);
    ggml_backend_tensor_set(q, Q.data(), 0, Q.size() * 4);
    ggml_backend_tensor_set(k, K.data(), 0, K.size() * 4);
    ggml_backend_tensor_set(v, V.data(), 0, V.size() * 4);
    t.run();
    auto r = t.get(packed);

    std::vector<double> want((size_t)D * S);
    std::vector<double> lg(L);
    for (int h = 0; h < H; h++)
        for (int n = 0; n < S; n++) {
            double smax = -1e30;
            for (int l = 0; l < L; l++) {
                double dot = 0;
                for (int dd = 0; dd < dh; dd++)
                    dot += (double)Q[(size_t)(h * dh + dd) + (size_t)n * D] * K[(size_t)(h * dh + dd) + (size_t)l * D];
                lg[l] = dot / std::sqrt((double)dh);
                smax = std::max(smax, lg[l]);
            }
            double den = 0;
            for (int l = 0; l < L; l++) { lg[l] = std::exp(lg[l] - smax); den += lg[l]; }
            for (int dd = 0; dd < dh; dd++) {
                double acc = 0;
                for (int l = 0; l < L; l++)
                    acc += lg[l] / den * (double)V[(size_t)(h * dh + dd) + (size_t)l * D];
                want[(size_t)dd + (size_t)h * dh + (size_t)n * D] = acc;
            }
        }
    check("flash_attn packed views", r, want, 1e-4);
}

// ---- 3. rope (the runtime's split formulation) vs the reference formula ----
static void test_rope() {
    const int dh = 64, T = 10;
    std::vector<float> X = randoms((size_t)dh * T), C = randoms((size_t)dh * T), Ss = randoms((size_t)dh * T);

    TestCtx t;
    t.ctx = ggml_init({1 << 24, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, dh, T);
    ggml_tensor* c = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, dh, T);
    ggml_tensor* s = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, dh, T);
    const int64_t h = dh / 2;
    const size_t es = 4;
    ggml_tensor* x1 = ggml_view_2d(t.ctx, x, h, T, x->nb[1], 0);
    ggml_tensor* x2 = ggml_view_2d(t.ctx, x, h, T, x->nb[1], h * es);
    ggml_tensor* c1 = ggml_view_2d(t.ctx, c, h, T, c->nb[1], 0);
    ggml_tensor* c2 = ggml_view_2d(t.ctx, c, h, T, c->nb[1], h * es);
    ggml_tensor* s1 = ggml_view_2d(t.ctx, s, h, T, s->nb[1], 0);
    ggml_tensor* s2 = ggml_view_2d(t.ctx, s, h, T, s->nb[1], h * es);
    ggml_tensor* out1 = ggml_sub(t.ctx, ggml_mul(t.ctx, x1, c1), ggml_mul(t.ctx, x2, s1));
    ggml_tensor* out2 = ggml_add(t.ctx, ggml_mul(t.ctx, x2, c2), ggml_mul(t.ctx, x1, s2));
    ggml_tensor* y = ggml_concat(t.ctx, out1, out2, 0);
    t.inputs = {x, c, s};
    t.alloc_graph(y, 4);
    ggml_backend_tensor_set(x, X.data(), 0, X.size() * 4);
    ggml_backend_tensor_set(c, C.data(), 0, C.size() * 4);
    ggml_backend_tensor_set(s, Ss.data(), 0, Ss.size() * 4);
    t.run();
    auto r = t.get(y);

    // reference: rot = cat(-x2, x1); out = x*c + rot*s
    std::vector<double> want((size_t)dh * T);
    for (int tt = 0; tt < T; tt++)
        for (int dd = 0; dd < dh; dd++) {
            double xc = (double)X[(size_t)dd + (size_t)tt * dh] * C[(size_t)dd + (size_t)tt * dh];
            double rot, sv = Ss[(size_t)dd + (size_t)tt * dh];
            if (dd < h) {
                rot = -(double)X[(size_t)dd + h + (size_t)tt * dh];
            } else {
                rot = (double)X[(size_t)dd - h + (size_t)tt * dh];
            }
            want[(size_t)dd + (size_t)tt * dh] = xc + rot * sv;
        }
    check("rope split formulation", r, want, 1e-5);
}

// ---- 4. L2 normalize along ne0 ----
static void test_l2norm() {
    const int D = 256, N = 8;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-3, 3);
    std::vector<float> X((size_t)D * N);
    for (auto& v : X) v = d(rng);

    TestCtx t;
    t.ctx = ggml_init({1 << 24, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, D, N);
    ggml_tensor* nrm = ggml_sqrt(t.ctx, ggml_sum_rows(t.ctx, ggml_sqr(t.ctx, x)));
    ggml_tensor* y = ggml_div(t.ctx, x, ggml_clamp(t.ctx, nrm, 1e-12f, 3.4e38f));
    t.inputs = {x};
    t.alloc_graph(y, 4);
    ggml_backend_tensor_set(x, X.data(), 0, X.size() * 4);
    t.run();
    auto r = t.get(y);

    std::vector<double> want((size_t)D * N);
    for (int n = 0; n < N; n++) {
        double acc = 0;
        for (int i = 0; i < D; i++) acc += (double)X[(size_t)i + (size_t)n * D] * X[(size_t)i + (size_t)n * D];
        double inv = 1.0 / std::sqrt(acc);
        for (int i = 0; i < D; i++) want[(size_t)i + (size_t)n * D] = X[(size_t)i + (size_t)n * D] * inv;
    }
    check("l2 normalize rows", r, want, 1e-5);
}

// ---- 5. bilinear x4 upsample vs a CPU reference (align_corners=False) ----
static void test_interpolate() {
    const int C = 4, fw = 8, ups = 4;
    std::vector<float> X = randoms((size_t)C * fw * fw);

    TestCtx t;
    t.ctx = ggml_init({1 << 26, nullptr, true});
    ggml_tensor* qf = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, fw, fw, C);
    ggml_tensor* up = ggml_interpolate(t.ctx, qf, fw * ups, fw * ups, C, 1, GGML_SCALE_MODE_BILINEAR);
    ggml_tensor* upc = ggml_cont(t.ctx, ggml_permute(t.ctx, up, 1, 2, 0, 3));  // {C, 4fw, 4fw}
    t.inputs = {qf};
    t.alloc_graph(upc, 4);
    ggml_backend_tensor_set(qf, X.data(), 0, X.size() * 4);
    t.run();
    auto r = t.get(upc);

    std::vector<double> want(r.size(), 0.0);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < fw * ups; y++)
            for (int x = 0; x < fw * ups; x++) {
                double sx = ((double)x + 0.5) / ups - 0.5;
                double sy = ((double)y + 0.5) / ups - 0.5;
                int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
                double fx = sx - x0, fy = sy - y0;
                auto px = [&](int xx, int yy) {
                    xx = std::min(std::max(xx, 0), fw - 1);
                    yy = std::min(std::max(yy, 0), fw - 1);
                    return (double)X[(size_t)c * fw * fw + (size_t)yy * fw + xx];
                };
                // upc ggml {C, H_up, W_up}: (c, x_up, y_up) at c + x_up*C + y_up*C*H_up
                want[(size_t)c + (size_t)x * C + (size_t)y * (size_t)C * fw * ups] =
                    px(x0, y0) * (1 - fx) * (1 - fy) + px(x0 + 1, y0) * fx * (1 - fy) +
                    px(x0, y0 + 1) * (1 - fx) * fy + px(x0 + 1, y0 + 1) * fx * fy;
            }
    check("bilinear upsample", r, want, 1e-4);
}

int main() {
    g_backend = ggml_backend_cpu_init();
    test_mul_mat();
    test_flash_attn();
    test_rope();
    test_l2norm();
    test_interpolate();
    printf("\n%s\n", g_failures ? "SOME TESTS FAILED" : "all op tests passed");
    return g_failures ? 1 : 0;
}
