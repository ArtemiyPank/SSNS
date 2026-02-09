#include <ssns/io/logger.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace ssns::io {

namespace {

// Round a single double to 4 decimal places.  std::round is half-away-from-
// zero; the Python reference uses torch.round (banker's rounding).  For the
// common case of 4-decimal weight values this difference only matters at
// values exactly on the half-tick (e.g. 0.12345 vs 0.12355) — well below
// the dz=0.09 confidence threshold the visualisation reasons about, so the
// discrepancy is documented and accepted.
constexpr int    kRoundDecimals = 4;
constexpr double kRoundScale    = 10000.0;   // 10^kRoundDecimals

inline double round4(double x) {
    return std::round(x * kRoundScale) / kRoundScale;
}

// Convert a 2D Matrix to a JSON nested array of 4-decimal-rounded values.
// Used for full weight tensors which are dense and large; the rounding
// keeps the resulting JSON tractable (~7 chars/float vs ~17 unrounded).
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

// Convert a single row of a 2D Matrix to a 1D JSON array of 4-decimal-
// rounded values.  Mirrors `_round_to_list_1d(t[i])` in the Python ref.
nlohmann::json matrix_row_to_rounded_1d(const ssns::linalg::Matrix& m,
                                        std::size_t row) {
    nlohmann::json arr = nlohmann::json::array();
    for (std::size_t c = 0; c < m.cols(); ++c) {
        arr.push_back(round4(m(row, c)));
    }
    return arr;
}

}  // namespace

TrainingLogger::TrainingLogger(LoggerConfig cfg) : cfg_(std::move(cfg)) {}

std::size_t TrainingLogger::snapshot_count() const noexcept {
    return snapshots_.size();
}

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
    // Cadence gate.  Python uses the same predicate; preserve it bit-for-bit
    // so a C++ run produces snapshot frames at the same epochs as the Python
    // reference for any given (epochs, snapshot_interval) pair.
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

    // Per-batch row capping: take the first min(samples_to_log, batch) rows.
    // batch is taken from H_raw.rows() (same as Python's H_raw.shape[0]).
    const std::size_t batch = H_raw.rows();
    const std::size_t cap   = (cfg_.samples_to_log < 0)
        ? 0
        : static_cast<std::size_t>(cfg_.samples_to_log);
    const std::size_t n     = (cap < batch) ? cap : batch;

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
        {"loss",    loss},        // full precision, NOT rounded
        {"lr",      lr},          // full precision, NOT rounded
        {"weights", std::move(weights)},
        {"samples", std::move(samples)},
    };
    snapshots_.push_back(std::move(snap));

    // Track the largest empirical n across recorded snapshots.  Equivalent
    // to taking the value from the most recent snapshot when batch size is
    // constant across training (always the case in our pipeline), but
    // robust to mixed-batch debug runs.
    if (n > samples_logged_) samples_logged_ = n;
}

void TrainingLogger::save(const std::filesystem::path& path) const {
    // Materialise metadata with the empirical samples_logged value rather
    // than the configured `samples_to_log`.  Python computes this lazily at
    // record time; we pick the equivalent rounded-up bound here so the
    // header always matches the actual content of `snapshots`.
    nlohmann::json metadata = {
        {"T_input",    cfg_.T_input},
        {"T_hidden",   cfg_.T_hidden},
        {"S_input",    cfg_.S_input},
        {"S_hidden",   cfg_.S_hidden},
        {"output_dim", cfg_.output_dim},
        {"cluster_size", cfg_.cluster_size},
        {"batch_size", cfg_.batch_size},
        {"epochs",     cfg_.epochs},
        {"dz",         cfg_.dz},                  // full precision
        {"lr_max",     cfg_.lr_max},              // full precision
        {"warmup_frac", cfg_.warmup_frac},        // full precision
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
            // Unindented dump — large weight matrices balloon 4–5x with
            // pretty-printing and the visualiser parses raw bytes anyway.
            out << payload.dump();
            out.flush();
            if (!out) {
                throw std::runtime_error(
                    "TrainingLogger::save: write failed: " + tmp.string());
            }
        }
        // POSIX `rename(2)` is atomic w.r.t. concurrent readers; on NTFS
        // modern std libs use MoveFileEx with replace semantics.
        std::filesystem::rename(tmp, path);
    } catch (...) {
        // Best-effort cleanup of the half-written tmp.  We never delete the
        // destination — concurrent readers stay safe with whatever was
        // there before.
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw;
    }
}

}  // namespace ssns::io
