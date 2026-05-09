#include <ssns/io/logger.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace ssns::io {

namespace {

// round to 4 decimals
// std::round is half away from zero python uses banker rounding
// half tick differences are way below dz=0.09 so fine
constexpr double kRoundScale = 10000.0;

// round x to 4 decimals
inline double round4(double x) {
    return std::round(x * kRoundScale) / kRoundScale;
}

// matrix to 2D json with 4 decimal rounding
// keeps json small for big weight tensors
nlohmann::json matrix_to_rounded_2d(const ssns::linalg::Matrix& m) {
    nlohmann::json arr = nlohmann::json::array();
    for (std::size_t r = 0; r < m.rows(); ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (std::size_t c = 0; c < m.cols(); ++c) {
            row.push_back(round4(m(r, c)));
        }
        arr.push_back(std::move(row));
    }
    return arr;
}

// one row to 1D json with rounding
nlohmann::json matrix_row_to_rounded_1d(const ssns::linalg::Matrix& m,
                                        std::size_t row) {
    nlohmann::json arr = nlohmann::json::array();
    for (std::size_t c = 0; c < m.cols(); ++c) {
        arr.push_back(round4(m(row, c)));
    }
    return arr;
}

}  // namespace

// store config snapshots start empty
TrainingLogger::TrainingLogger(LoggerConfig cfg) : cfg_(std::move(cfg)) {}

// number of snapshots
std::size_t TrainingLogger::snapshot_count() const noexcept {
    return snapshots_.size();
}

// record one snapshot if epoch matches cadence
void TrainingLogger::maybe_record(
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
    const ssns::linalg::Matrix& error)
{
    // cadence gate same as python ref
    if (epoch <= 0) return;
    if (cfg_.snapshot_interval <= 0) return;
    if (epoch % cfg_.snapshot_interval != 0) return;

    nlohmann::json weights = {
        {"W1_T", matrix_to_rounded_2d(W1_T)},
        {"W2_T", matrix_to_rounded_2d(W2_T)},
        {"W1",   matrix_to_rounded_2d(W1)},
        {"W2",   matrix_to_rounded_2d(W2)},
        {"B_FA", matrix_to_rounded_2d(B_FA)},
    };

    // first min(samples_to_log, batch) rows
    const std::size_t batch = H_raw.rows();
    const std::size_t cap = (cfg_.samples_to_log < 0)
        ? 0
        : static_cast<std::size_t>(cfg_.samples_to_log);
    const std::size_t n = (cap < batch) ? cap : batch;

    nlohmann::json samples = nlohmann::json::array();
    for (std::size_t i = 0; i < n; ++i) {
        samples.push_back({
            {"X",      matrix_row_to_rounded_1d(X,      i)},
            {"H_T",    matrix_row_to_rounded_1d(H_T,    i)},
            {"H_raw",  matrix_row_to_rounded_1d(H_raw,  i)},
            {"Y_true", matrix_row_to_rounded_1d(Y_true, i)},
            {"Y_pred", matrix_row_to_rounded_1d(Y_pred, i)},
            {"error",  matrix_row_to_rounded_1d(error,  i)},
        });
    }

    nlohmann::json snap = {
        {"epoch",   epoch},
        {"loss",    loss},        // not rounded
        {"lr",      lr},          // not rounded
        {"weights", std::move(weights)},
        {"samples", std::move(samples)},
    };
    snapshots_.push_back(std::move(snap));

    // track largest n seen
    if (n > samples_logged_) samples_logged_ = n;
}

// atomic write of all snapshots plus metadata
void TrainingLogger::save(const std::filesystem::path& path) const {
    // metadata uses empirical samples_logged
    nlohmann::json metadata = {
        {"T_input",    cfg_.T_input},
        {"T_hidden",   cfg_.T_hidden},
        {"S_input",    cfg_.S_input},
        {"S_hidden",   cfg_.S_hidden},
        {"output_dim", cfg_.output_dim},
        {"cluster_size", cfg_.cluster_size},
        {"batch_size", cfg_.batch_size},
        {"epochs",     cfg_.epochs},
        {"dz",         cfg_.dz},
        {"lr_max",     cfg_.lr_max},
        {"warmup_frac", cfg_.warmup_frac},
        {"snapshot_interval", cfg_.snapshot_interval},
        {"samples_logged",    samples_logged_},
        {"teacher_seed", cfg_.teacher_seed},
        {"bfa_seed",     cfg_.bfa_seed},
    };

    nlohmann::json payload = {
        {"metadata",  std::move(metadata)},
        {"snapshots", snapshots_},
    };

    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    auto tmp = path;
    tmp += ".tmp";
    try {
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error(
                    "TrainingLogger::save: failed to open tmp file: "
                    + tmp.string());
            }
            // unindented dump pretty print blows up file size
            out << payload.dump();
            out.flush();
            if (!out) {
                throw std::runtime_error(
                    "TrainingLogger::save: write failed: " + tmp.string());
            }
        }
        // rename is atomic on posix and ntfs
        std::filesystem::rename(tmp, path);
    } catch (...) {
        // best effort cleanup of tmp
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw;
    }
}

}  // namespace ssns::io
