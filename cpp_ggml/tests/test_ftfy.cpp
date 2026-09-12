// String-level comparison: engine ftfy subset vs the real ftfy (ground truth).
#include "../src/tokenizer.cpp"   // white-box: access static ftfy_subset
#include "ftfy_cases.inc"          // ground truth generated from real ftfy
#include <cstdio>
#include <string>
#include <vector>

int main() {
    using namespace gkd;
    int pass = 0, fail = 0;
    std::string report;
    for (int i = 0; i < ftfy_cases_n; i++) {
        std::string got = ftfy_cases[i].in;
        gkd::ftfy_subset(got);
        bool ok = (got == ftfy_cases[i].want);
        if (ok) pass++; else fail++;
        std::string in_s = ftfy_cases[i].in, want_s = ftfy_cases[i].want;
        for (auto& str : {&in_s, &want_s, &got}) {}  // keep raw
        report += std::string(ok ? "OK  " : "FAIL") + " in=" + in_s +
                  " want=" + want_s + " got=" + got + "\n";
    }
    char tail[64];
    snprintf(tail, sizeof(tail), "pass=%d fail=%d", pass, fail);
    report += tail;
    printf("%s\n", report.c_str());
    return fail ? 1 : 0;
}
