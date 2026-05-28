// io::TrainingLogger writes ui_data/training_log.json with the exact schema the Python frontend reads
// schema mirrors SSNS_mvp/src/ssns_clean/logger.py
//
// numerical rules
//   weights[*] and samples[*][...] rounded to 4 decimals
//   scalars (loss, lr, dz, lr_max, warmup_frac) keep full precision
//   epoch <= 0 is NOT recorded
//   snapshot recorded iff (epoch >= 1) AND (epoch % snapshot_interval == 0)
#include <catch.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include <ssns/io/logger.hpp>
#include <ssns/linalg/matrix.hpp>

using ssns::io::LoggerConfig;
using ssns::io::TrainingLogger;
using ssns::linalg::Matrix;
using nlohmann::json;

namespace {

// fresh temp dir under /tmp, removed if it already exists
std::filesystem::path tmp_dir(const char* tag) {
    auto dir = std::filesystem::temp_directory_path()
             / ("ssns_logger_test_" + std::string(tag));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// constant-filled matrix for deterministic test setup
Matrix filled(std::size_t rows, std::size_t cols, double value) {
    return Matrix::full(rows, cols, value);
}

// compact test config, toy dims with real-shape geometry
LoggerConfig make_cfg(long interval = 100, long samples = 4) {
    LoggerConfig c;
    c.snapshot_interval = interval;
    c.samples_to_log    = samples;
    c.T_input    = 4;
    c.T_hidden   = 6;
    c.S_input    = 4;
    c.S_hidden   = 8;
    c.output_dim = 10;     // key_bits * cluster_size, here 5 * 2 just for shape
    c.cluster_size = 2;
    c.batch_size = 8;
    c.epochs     = 1000;
    c.dz         = 0.09;
    c.lr_max     = 0.01;
    c.warmup_frac = 0.05;
    c.teacher_seed = 42;
    c.bfa_seed     = 7;
    return c;
}

// full set of tensors with canonical shapes for the LoggerConfig
// each tensor filled with `value`, batch row count is per-call
struct Tensors {
    Matrix W1_T, W2_T, W1, W2, B_FA;
    Matrix X, H_T, H_raw, Y_true, Y_pred, error;
};

// build canonical-shape tensors for a config, all filled with v
Tensors make_tensors(const LoggerConfig& c, std::size_t batch, double v) {
    return Tensors{
        filled(c.T_input,    c.T_hidden,   v),
        filled(c.T_hidden,   c.output_dim, v),
        filled(c.S_input,    c.S_hidden,   v),
        filled(c.S_hidden,   c.output_dim, v),
        filled(c.output_dim, c.S_hidden,   v),
        filled(batch, c.T_input,    v),
        filled(batch, c.T_hidden,   v),
        filled(batch, c.S_hidden,   v),
        filled(batch, c.output_dim, v),
        filled(batch, c.output_dim, v),
        filled(batch, c.output_dim, v),
    };
}

}  // namespace

TEST_CASE("Logger: snapshot interval respected - epoch 99 with interval 100 skipped",
          "[io][logger]") {
    TrainingLogger lg(make_cfg(/*interval=*/100));
    auto t = make_tensors(make_cfg(), /*batch=*/8, /*v=*/0.5);
    lg.maybe_record(99, /*loss=*/0.1, /*lr=*/0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);
    REQUIRE(lg.snapshot_count() == 0);
}

TEST_CASE("Logger: snapshot interval respected - epoch 100 with interval 100 records",
          "[io][logger]") {
    TrainingLogger lg(make_cfg(/*interval=*/100));
    auto t = make_tensors(make_cfg(), /*batch=*/8, /*v=*/0.5);
    lg.maybe_record(100, 0.1, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);
    REQUIRE(lg.snapshot_count() == 1);
}

TEST_CASE("Logger: epoch=0 is not recorded", "[io][logger]") {
    TrainingLogger lg(make_cfg(/*interval=*/1));    // would match if epoch>=1
    auto t = make_tensors(make_cfg(), /*batch=*/8, /*v=*/0.5);
    lg.maybe_record(0, 0.1, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);
    REQUIRE(lg.snapshot_count() == 0);
}

TEST_CASE("Logger: negative epoch is not recorded", "[io][logger]") {
    TrainingLogger lg(make_cfg(/*interval=*/1));
    auto t = make_tensors(make_cfg(), /*batch=*/8, /*v=*/0.5);
    lg.maybe_record(-5, 0.1, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);
    REQUIRE(lg.snapshot_count() == 0);
}

TEST_CASE("Logger: only first samples_to_log batch rows are saved",
          "[io][logger]") {
    auto cfg = make_cfg(/*interval=*/1, /*samples=*/3);
    TrainingLogger lg(cfg);
    auto t = make_tensors(cfg, /*batch=*/8, /*v=*/0.5);
    lg.maybe_record(1, 0.1, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);

    auto path = tmp_dir("samples_clip") / "training_log.json";
    lg.save(path);
    std::ifstream in(path); json j; in >> j;
    REQUIRE(j["snapshots"].size() == 1);
    REQUIRE(j["snapshots"][0]["samples"].size() == 3);  // capped at samples_to_log
    REQUIRE(j["snapshots"][0]["samples"][0]["X"].size() == cfg.T_input);
    REQUIRE(j["snapshots"][0]["samples"][0]["H_T"].size() == cfg.T_hidden);
    REQUIRE(j["snapshots"][0]["samples"][0]["H_raw"].size() == cfg.S_hidden);
    REQUIRE(j["snapshots"][0]["samples"][0]["Y_true"].size() == cfg.output_dim);
    REQUIRE(j["snapshots"][0]["samples"][0]["Y_pred"].size() == cfg.output_dim);
    REQUIRE(j["snapshots"][0]["samples"][0]["error"].size() == cfg.output_dim);
}

TEST_CASE("Logger: 4-decimal rounding on weight tensors", "[io][logger]") {
    auto cfg = make_cfg(/*interval=*/1, /*samples=*/1);
    TrainingLogger lg(cfg);
    // 0.123456 -> 0.1235 (both round-half-to-even and std::round half-away-from-zero
    // produce 0.1235, the 5th decimal is 5 with a 6 trailing)
    auto t = make_tensors(cfg, /*batch=*/2, /*v=*/0.123456);
    lg.maybe_record(1, 0.1, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);

    auto path = tmp_dir("rounding") / "training_log.json";
    lg.save(path);
    std::ifstream in(path); json j; in >> j;

    const auto v = j["snapshots"][0]["weights"]["W1"][0][0].get<double>();
    REQUIRE(v == Approx(0.1235).epsilon(1e-9));

    // spot-check sample data (1D path) too
    const auto x = j["snapshots"][0]["samples"][0]["X"][0].get<double>();
    REQUIRE(x == Approx(0.1235).epsilon(1e-9));
}

TEST_CASE("Logger: loss and lr keep full precision (1e-8 not zeroed)",
          "[io][logger]") {
    auto cfg = make_cfg(/*interval=*/1);
    TrainingLogger lg(cfg);
    auto t = make_tensors(cfg, /*batch=*/2, /*v=*/0.5);
    const double tiny_loss = 1.234567890e-8;
    const double tiny_lr   = 9.87654321e-9;
    lg.maybe_record(1, tiny_loss, tiny_lr,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);

    auto path = tmp_dir("precision") / "training_log.json";
    lg.save(path);
    std::ifstream in(path); json j; in >> j;
    REQUIRE(j["snapshots"][0]["loss"].get<double>() == Approx(tiny_loss).epsilon(1e-15));
    REQUIRE(j["snapshots"][0]["lr"].get<double>()   == Approx(tiny_lr).epsilon(1e-15));
}

TEST_CASE("Logger: samples_logged metadata = min(samples_to_log, actual batch)",
          "[io][logger]") {
    // case A: samples_to_log < batch -> metadata = samples_to_log
    auto cfg_a = make_cfg(/*interval=*/1, /*samples=*/3);
    TrainingLogger lg_a(cfg_a);
    auto t_a = make_tensors(cfg_a, /*batch=*/8, /*v=*/0.5);
    lg_a.maybe_record(1, 0.1, 0.01,
                      t_a.W1_T, t_a.W2_T, t_a.W1, t_a.W2, t_a.B_FA,
                      t_a.X, t_a.H_T, t_a.H_raw, t_a.Y_true, t_a.Y_pred, t_a.error);
    auto path_a = tmp_dir("meta_clip_a") / "training_log.json";
    lg_a.save(path_a);
    std::ifstream in_a(path_a); json j_a; in_a >> j_a;
    REQUIRE(j_a["metadata"]["samples_logged"] == 3);

    // case B: samples_to_log > batch -> metadata = batch
    auto cfg_b = make_cfg(/*interval=*/1, /*samples=*/100);
    TrainingLogger lg_b(cfg_b);
    auto t_b = make_tensors(cfg_b, /*batch=*/5, /*v=*/0.5);
    lg_b.maybe_record(1, 0.1, 0.01,
                      t_b.W1_T, t_b.W2_T, t_b.W1, t_b.W2, t_b.B_FA,
                      t_b.X, t_b.H_T, t_b.H_raw, t_b.Y_true, t_b.Y_pred, t_b.error);
    auto path_b = tmp_dir("meta_clip_b") / "training_log.json";
    lg_b.save(path_b);
    std::ifstream in_b(path_b); json j_b; in_b >> j_b;
    REQUIRE(j_b["metadata"]["samples_logged"] == 5);
}

TEST_CASE("Logger: save round-trip - required keys + shapes match",
          "[io][logger]") {
    auto cfg = make_cfg(/*interval=*/100, /*samples=*/2);
    TrainingLogger lg(cfg);
    auto t = make_tensors(cfg, /*batch=*/8, /*v=*/0.25);
    lg.maybe_record(100, 0.5, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);
    lg.maybe_record(200, 0.4, 0.009,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);

    auto path = tmp_dir("roundtrip") / "training_log.json";
    lg.save(path);
    REQUIRE(std::filesystem::exists(path));

    std::ifstream in(path); json j; in >> j;

    // top-level keys
    REQUIRE(j.contains("metadata"));
    REQUIRE(j.contains("snapshots"));

    // metadata fields
    const auto& m = j["metadata"];
    REQUIRE(m["T_input"]    == cfg.T_input);
    REQUIRE(m["T_hidden"]   == cfg.T_hidden);
    REQUIRE(m["S_input"]    == cfg.S_input);
    REQUIRE(m["S_hidden"]   == cfg.S_hidden);
    REQUIRE(m["output_dim"] == cfg.output_dim);
    REQUIRE(m["cluster_size"] == cfg.cluster_size);
    REQUIRE(m["batch_size"] == cfg.batch_size);
    REQUIRE(m["epochs"]     == cfg.epochs);
    REQUIRE(m["dz"].get<double>()          == Approx(cfg.dz));
    REQUIRE(m["lr_max"].get<double>()      == Approx(cfg.lr_max));
    REQUIRE(m["warmup_frac"].get<double>() == Approx(cfg.warmup_frac));
    REQUIRE(m["snapshot_interval"] == cfg.snapshot_interval);
    REQUIRE(m["samples_logged"]   == 2);     // min(samples_to_log=2, batch=8)
    REQUIRE(m["teacher_seed"]     == cfg.teacher_seed);
    REQUIRE(m["bfa_seed"]         == cfg.bfa_seed);

    // two snapshots
    REQUIRE(j["snapshots"].size() == 2);

    // first snapshot shape
    const auto& s0 = j["snapshots"][0];
    REQUIRE(s0["epoch"] == 100);
    REQUIRE(s0["loss"].get<double>() == Approx(0.5));
    REQUIRE(s0["lr"].get<double>()   == Approx(0.01));

    // weight matrix shapes (rows = outer list size, cols = inner list size)
    const auto& w = s0["weights"];
    REQUIRE(w["W1_T"].size() == cfg.T_input);
    REQUIRE(w["W1_T"][0].size() == cfg.T_hidden);
    REQUIRE(w["W2_T"].size() == cfg.T_hidden);
    REQUIRE(w["W2_T"][0].size() == cfg.output_dim);
    REQUIRE(w["W1"].size() == cfg.S_input);
    REQUIRE(w["W1"][0].size() == cfg.S_hidden);
    REQUIRE(w["W2"].size() == cfg.S_hidden);
    REQUIRE(w["W2"][0].size() == cfg.output_dim);
    REQUIRE(w["B_FA"].size() == cfg.output_dim);
    REQUIRE(w["B_FA"][0].size() == cfg.S_hidden);

    // samples shape
    REQUIRE(s0["samples"].size() == 2);
    REQUIRE(s0["samples"][0]["X"].size()      == cfg.T_input);
    REQUIRE(s0["samples"][0]["H_T"].size()    == cfg.T_hidden);
    REQUIRE(s0["samples"][0]["H_raw"].size()  == cfg.S_hidden);
    REQUIRE(s0["samples"][0]["Y_true"].size() == cfg.output_dim);
    REQUIRE(s0["samples"][0]["Y_pred"].size() == cfg.output_dim);
    REQUIRE(s0["samples"][0]["error"].size()  == cfg.output_dim);
}

TEST_CASE("Logger: atomic save - no .tmp leftover after successful save",
          "[io][logger]") {
    auto cfg = make_cfg(/*interval=*/1);
    TrainingLogger lg(cfg);
    auto t = make_tensors(cfg, /*batch=*/2, /*v=*/0.5);
    lg.maybe_record(1, 0.1, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);

    auto dir  = tmp_dir("atomic");
    auto path = dir / "training_log.json";
    lg.save(path);

    bool any_tmp = false;
    for (auto& p : std::filesystem::directory_iterator(dir)) {
        if (p.path().extension() == ".tmp") { any_tmp = true; break; }
    }
    REQUIRE_FALSE(any_tmp);
    REQUIRE(std::filesystem::exists(path));
}

TEST_CASE("Logger: save creates parent directories if missing",
          "[io][logger]") {
    auto cfg = make_cfg(/*interval=*/1);
    TrainingLogger lg(cfg);
    auto t = make_tensors(cfg, /*batch=*/2, /*v=*/0.5);
    lg.maybe_record(1, 0.1, 0.01,
                    t.W1_T, t.W2_T, t.W1, t.W2, t.B_FA,
                    t.X, t.H_T, t.H_raw, t.Y_true, t.Y_pred, t.error);

    auto root = std::filesystem::temp_directory_path() / "ssns_logger_test_parents";
    std::filesystem::remove_all(root);
    auto path = root / "deep" / "nested" / "training_log.json";
    lg.save(path);
    REQUIRE(std::filesystem::exists(path));
}
