// training trajectory logger
// writes ui_data/training_log.json for the IDE epoch slider
// mirrors SSNS_mvp/src/ssns_clean/logger.py
//
// json schema:
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
//                 "loss":  float,    // not rounded
//                 "lr":    float,    // not rounded
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
// rounding tensors to 4 decimals to keep json small
// loss and lr full precision because lr can go to 1e-9 at end of cosine
//
// atomic save write tmp then rename
#ifndef SSNS_IO_LOGGER_HPP
#define SSNS_IO_LOGGER_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

#include <ssns/linalg/matrix.hpp>

namespace ssns::io {

// config for one logger run
struct LoggerConfig {
    long snapshot_interval;
    long samples_to_log;

    // arch and training metadata recorded once
    std::size_t T_input, T_hidden, S_input, S_hidden;
    std::size_t output_dim, cluster_size;
    std::size_t batch_size;
    long epochs;
    double dz, lr_max, warmup_frac;
    std::uint64_t teacher_seed, bfa_seed;
};

// records snapshots and writes them as one json file
class TrainingLogger {
public:
    // build logger that accepts snapshots from cfg
    explicit TrainingLogger(LoggerConfig cfg);

    // append snapshot if epoch >= 1 and epoch % snapshot_interval == 0
    // tensor shapes:
    //   W1_T  [T_input, T_hidden]    teacher layer 1 frozen
    //   W2_T  [T_hidden, output_dim] teacher layer 2 frozen
    //   W1    [S_input, S_hidden]    student layer 1
    //   W2    [S_hidden, output_dim] student layer 2
    //   B_FA  [output_dim, S_hidden] feedback alignment
    //   X     [batch, T_input]
    //   H_T   [batch, T_hidden]
    //   H_raw [batch, S_hidden]
    //   Y_true, Y_pred, error  [batch, output_dim]
    // only first min(samples_to_log, batch) rows stored
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

    // atomic save write tmp then rename
    // creates parent dirs if needed
    void save(const std::filesystem::path& path) const;

    // snapshots in memory
    [[nodiscard]] std::size_t snapshot_count() const noexcept;

private:
    LoggerConfig cfg_;
    // snapshots accumulate as json fragments
    std::vector<nlohmann::json> snapshots_;
    // largest sample count actually written
    std::size_t samples_logged_ = 0;
};

}  // namespace ssns::io

#endif  // SSNS_IO_LOGGER_HPP
