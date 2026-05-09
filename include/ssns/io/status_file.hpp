// small status json written by training subprocess polled by IDE every 500 ms
// schema matches scripts/benchmark_fhe_vs_plain.py and app.py
//
// schema:
//   {
//     "started_at":   "2026-04-29T13:45:00Z",
//     "epoch":        int,
//     "total_epochs": int,
//     "loss":         float,
//     "lr":           float,
//     "elapsed_sec":  float,
//     "eta_sec":      float | null,
//     "running":      bool,
//     "completed_at": "ISO8601 string" | null
//   }
//
// all writes are atomic write tmp then rename
#ifndef SSNS_IO_STATUS_FILE_HPP
#define SSNS_IO_STATUS_FILE_HPP

#include <chrono>
#include <filesystem>
#include <optional>

namespace ssns::io {

using SystemTimePoint = std::chrono::system_clock::time_point;

// initial placeholder running=true epoch=0 completed_at=null
// call on /api/run_training entry to clear stale state
void write_starting_status(
    const std::filesystem::path& path,
    long total_epochs);

// full progress record
// started_at lets us compute elapsed_sec and eta_sec
// pass nullopt for completed_at while running real time at end
void write_progress(
    const std::filesystem::path& path,
    long epoch,
    long total_epochs,
    double loss,
    double lr,
    SystemTimePoint started_at,
    bool running,
    std::optional<SystemTimePoint> completed_at);

}  // namespace ssns::io

#endif  // SSNS_IO_STATUS_FILE_HPP
