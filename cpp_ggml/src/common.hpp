// GKDT ggml runtime - common utilities.
#pragma once

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace gkd {

enum class LogLevel { Debug = 0, Info, Warn, Error };

void set_log_level(LogLevel level);
void logf(LogLevel level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

#define GKD_LOG_DEBUG(...) ::gkd::logf(::gkd::LogLevel::Debug, __VA_ARGS__)
#define GKD_LOG_INFO(...)  ::gkd::logf(::gkd::LogLevel::Info,  __VA_ARGS__)
#define GKD_LOG_WARN(...)  ::gkd::logf(::gkd::LogLevel::Warn,  __VA_ARGS__)
#define GKD_LOG_ERROR(...) ::gkd::logf(::gkd::LogLevel::Error, __VA_ARGS__)

// Monotonic wall clock in milliseconds.
double now_ms();

// Simple steady-state statistics.
struct TimingStats {
    int    count = 0;
    double total_ms = 0.0;
    double min_ms = 1e30;
    double max_ms = 0.0;
    std::vector<double> samples;  // kept only when record_samples=true

    void add(double ms, bool record_samples = false);
    double mean_ms() const { return count > 0 ? total_ms / count : 0.0; }
    double p(int percent) const;  // percentile over samples (requires record_samples)
};

// Dump a float32 array as `[int64 ndims, int64 dim...] + row-major data`.
void dump_f32(const std::string& path, const std::vector<int64_t>& shape, const float* data);

// A simple single-header JSON string escaper for bench records.
std::string json_escape(const std::string& s);

}  // namespace gkd
