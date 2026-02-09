// Phase-3 training-trajectory logger.  Produces ui_data/training_log.json
// in the exact schema consumed by the Python reference frontend (ui/) and
// the Visual IDE epoch slider.  Mirrors the reference Python implementation
// at SSNS_mvp/src/ssns_clean/logger.py.
//
// JSON schema (top level):
//
//     {
//         "metadata": {
//             "T_input": int, "T_hidden": int, "S_input": int, "S_hidden": int,
//             "output_dim": int, "cluster_size": int, "batch_size": int,
//             "epochs": int, "dz": float, "lr_max": float, "warmup_frac": float,
//             "snapshot_interval": int, "samples_logged": int,
//             "teacher_seed": int, "bfa_seed": int
//         },
//         "snapshots": [
//             {
//                 "epoch": int,
//                 "loss":  float,    // FULL precision — NOT rounded.
//                 "lr":    float,    // FULL precision — NOT rounded.
//                 "weights": {
//                     "W1_T": [[float]], "W2_T": [[float]],
//                     "W1":   [[float]], "W2":   [[float]],
//                     "B_FA": [[float]]
//                 },
//                 "samples": [
//                     {
//                         "X": [float], "H_T": [float], "H_raw": [float],
//                         "Y_true": [float], "Y_pred": [float], "error": [float]
//                     }, ...
//                 ]
//             }, ...
//         ]
//     }
//
// Numerical asymmetry: weight & sample tensor data is rounded to 4 decimals
// (keeps JSON tractable for large networks) while loss/lr keep full
// precision (their dynamic range — e.g. 1e-9 lr at end of cosine — would
// zero out at 4 decimals).
//
// Atomic save: writes to <path>.tmp and renames onto <path>; concurrent
// readers always see either the previous complete file or the new one.
#ifndef SSNS_IO_LOGGER_HPP
#define SSNS_IO_LOGGER_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

#include <ssns/linalg/matrix.hpp>

namespace ssns::io {

struct LoggerConfig {
    long snapshot_interval;
    long samples_to_log;

    // Architecture / training metadata recorded once in the file header.
    std::size_t T_input, T_hidden, S_input, S_hidden;
    std::size_t output_dim, cluster_size;
    std::size_t batch_size;
    long epochs;
    double dz, lr_max, warmup_frac;
    std::uint64_t teacher_seed, bfa_seed;
};

class TrainingLogger {
public:
    explicit TrainingLogger(LoggerConfig cfg);

    // Snapshot is appended iff (epoch >= 1) AND (epoch % snapshot_interval == 0).
    // Tensors named per the Python schema:
    //   W1_T  [T_input, T_hidden]    (Teacher layer 1, frozen)
    //   W2_T  [T_hidden, output_dim] (Teacher layer 2, frozen)
    //   W1    [S_input, S_hidden]    (Student layer 1)
    //   W2    [S_hidden, output_dim] (Student layer 2)
    //   B_FA  [output_dim, S_hidden] (Server feedback alignment)
    //   X     [batch, T_input]
    //   H_T   [batch, T_hidden]
    //   H_raw [batch, S_hidden]
    //   Y_true, Y_pred, error  [batch, output_dim]
    // Only the FIRST min(samples_to_log, batch) rows of the per-batch tensors
    // are stored.
    void maybe_record(
        long epoch, double loss, double lr,
        const ssns::linalg::Matrix& W1_T,
        const ssns::linalg::Matrix& W2_T,
        const ssns::linalg::Matrix& W1,
        const ssns::linalg::Matrix& W2,
        const ssns::linalg::Matrix& B_FA,
        const ssns::linalg::Matrix& X,
        const ssns::linalg::Matrix& H_T,
        const ssns::linalg::Matrix& H_raw,
        const ssns::linalg::Matrix& Y_true,
        const ssns::linalg::Matrix& Y_pred,
        const ssns::linalg::Matrix& error);

    // Atomic write: <path>.tmp + std::filesystem::rename, plus best-effort
    // cleanup of the tmp file on exception so a half-written sibling is
    // never left behind.  Creates parent directories if missing.
    void save(const std::filesystem::path& path) const;

    [[nodiscard]] std::size_t snapshot_count() const noexcept;

private:
    LoggerConfig cfg_;
    // Snapshots accumulate as nlohmann::json fragments — building the JSON
    // payload incrementally avoids a second traversal at save() time and
    // keeps each maybe_record() call O(snapshot_size).
    std::vector<nlohmann::json> snapshots_;
    // Effective number of samples actually logged in the snapshots already
    // recorded.  Used to populate metadata.samples_logged at save time so
    // it reflects the empirical min(samples_to_log, batch_size) seen.
    std::size_t samples_logged_ = 0;
};

}  // namespace ssns::io

#endif  // SSNS_IO_LOGGER_HPP
