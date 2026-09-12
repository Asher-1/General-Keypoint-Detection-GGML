// GKDT ggml runtime - common utilities.
#include "common.hpp"

#include <cstdarg>
#include <cstdlib>
#include <filesystem>
#include <algorithm>

namespace gkd {

static LogLevel g_level = LogLevel::Info;

void set_log_level(LogLevel level) { g_level = level; }

void logf(LogLevel level, const char* fmt, ...) {
    if (level < g_level) return;
    const char* tag = "[GKD]";
    switch (level) {
        case LogLevel::Debug: tag = "[GKD:dbg]"; break;
        case LogLevel::Info:  tag = "[GKD]"; break;
        case LogLevel::Warn:  tag = "[GKD:warn]"; break;
        case LogLevel::Error: tag = "[GKD:err]"; break;
    }
    std::fprintf(stderr, "%s ", tag);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

void TimingStats::add(double ms, bool record_samples) {
    count++;
    total_ms += ms;
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
    if (record_samples) samples.push_back(ms);
}

double TimingStats::p(int percent) const {
    if (samples.empty()) return max_ms;
    std::vector<double> s = samples;
    std::sort(s.begin(), s.end());
    size_t idx = std::min(s.size() - 1, (size_t)((percent / 100.0) * s.size()));
    return s[idx];
}

void dump_f32(const std::string& path, const std::vector<int64_t>& shape, const float* data) {
    // create parent directories so --dump-taps works with a fresh path
    size_t slash = path.rfind('/');
    if (slash != std::string::npos) {
        std::error_code ec;
        std::filesystem::create_directories(path.substr(0, slash), ec);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        GKD_LOG_ERROR("cannot open %s for writing", path.c_str());
        return;
    }
    int64_t ndims = (int64_t)shape.size();
    std::fwrite(&ndims, sizeof(int64_t), 1, f);
    std::fwrite(shape.data(), sizeof(int64_t), shape.size(), f);
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

}  // namespace gkd
