#include <ssns/io/status_file.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ssns::io {

namespace {

// Format a system_clock::time_point as ISO 8601 UTC with second precision,
// e.g. "2026-04-29T13:45:00Z".  Uses gmtime_r / strftime — std::format's
// chrono support is patchy across compilers in 2026 still.
std::string iso8601_utc(SystemTimePoint tp) {
    const auto t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

void atomic_write_json(const std::filesystem::path& path,
                       const nlohmann::json& payload) {
    std::filesystem::create_directories(path.parent_path());
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << payload.dump();
        out.flush();
    }
    // std::filesystem::rename is required by the standard to be atomic on
    // POSIX (effectively rename(2)).  On NTFS modern std libs use
    // MoveFileEx; semantics are good enough for our IDE poller.
    std::filesystem::rename(tmp, path);
}

}  // namespace

void write_starting_status(const std::filesystem::path& path, long total_epochs) {
    const auto now = std::chrono::system_clock::now();
    nlohmann::json j = {
        {"started_at",   iso8601_utc(now)},
        {"epoch",        0},
        {"total_epochs", total_epochs},
        {"loss",         0.0},
        {"lr",           0.0},
        {"elapsed_sec",  0.0},
        {"eta_sec",      nullptr},
        {"running",      true},
        {"completed_at", nullptr},
    };
    atomic_write_json(path, j);
}

void write_progress(
    const std::filesystem::path& path,
    long epoch, long total_epochs,
    double loss, double lr,
    SystemTimePoint started_at,
    bool running,
    std::optional<SystemTimePoint> completed_at)
{
    const auto now = std::chrono::system_clock::now();
    const double elapsed = std::chrono::duration<double>(now - started_at).count();

    std::optional<double> eta_sec;
    if (running && epoch > 0) {
        const double rate = elapsed / static_cast<double>(epoch);
        const double remaining = static_cast<double>(total_epochs - epoch);
        eta_sec = (remaining > 0.0) ? rate * remaining : 0.0;
    }

    nlohmann::json j = {
        {"started_at",   iso8601_utc(started_at)},
        {"epoch",        epoch},
        {"total_epochs", total_epochs},
        {"loss",         loss},
        {"lr",           lr},
        {"elapsed_sec",  elapsed},
        {"eta_sec",      eta_sec.has_value() ? nlohmann::json(*eta_sec) : nlohmann::json(nullptr)},
        {"running",      running},
        {"completed_at", completed_at.has_value()
                            ? nlohmann::json(iso8601_utc(*completed_at))
                            : nlohmann::json(nullptr)},
    };
    atomic_write_json(path, j);
}

}  // namespace ssns::io
