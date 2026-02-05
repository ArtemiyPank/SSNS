// Lightweight status JSON written by the training subprocess and polled by
// the IDE frontend every ~500 ms.  Mirrors the schema produced by Python's
// scripts/benchmark_fhe_vs_plain.py and app.py:_write_starting_status.
//
// Schema:
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
// Always written atomically (write to <path>.tmp + rename).
#ifndef SSNS_IO_STATUS_FILE_HPP
#define SSNS_IO_STATUS_FILE_HPP

#include <chrono>
#include <filesystem>
#include <optional>

namespace ssns::io {

using SystemTimePoint = std::chrono::system_clock::time_point;

// Initial placeholder: running=true, epoch=0, completed_at=null.
// Use this on /api/run_training entry to wipe any stale state from prior
// runs, before the subprocess writes its first real status update.
void write_starting_status(
    const std::filesystem::path& path,
    long total_epochs);

// Full progress write.  `started_at` lets the function compute elapsed_sec
// and eta_sec (when running and epoch > 0).  Pass nullopt for completed_at
// during a running step; pass a real time at training end.
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
