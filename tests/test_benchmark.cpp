// integration test for ssns-benchmark CLI (subprocess spawned by HTTP server /api/run_training)
// drives a tiny 50-epoch run and checks output JSON schema
#include <catch.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using nlohmann::json;

static fs::path tmp_dir(const char* tag) {
    auto dir = fs::temp_directory_path() / ("ssns_bench_test_" + std::string(tag));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

TEST_CASE("benchmark CLI: tiny run produces schema-correct logs",
          "[benchmark][slow]") {
    auto dir = tmp_dir("tiny");
    auto log_path    = dir / "training_log.json";
    auto status_path = dir / "training_status.json";

    // CMake places the binary next to ssns_tests; prefer env var, fall back to sibling
    fs::path bench;
    if (const char* env = std::getenv("SSNS_BENCHMARK_BINARY")) {
        bench = env;
    } else {
        // ctest runs from build/, binary lives here
        bench = "./ssns-benchmark";
        if (!fs::exists(bench)) bench = "build/ssns-benchmark";
    }
    REQUIRE(fs::exists(bench));

    std::string cmd = bench.string()
        + " --t-input 4 --t-hidden 4 --s-input 4 --s-hidden 8"
        + " --output-dim 10 --cluster-size 5"
        + " --batch-size 8 --epochs 50 --snapshot-interval 10"
        + " --samples-to-log 2"
        + " --dz 0.10 --lr-max 0.01 --warmup-frac 0.05"
        + " --teacher-seed 42 --bfa-seed 43 --seed 2024"
        + " --output-path " + log_path.string()
        + " --status-path " + status_path.string();

    const int rc = std::system(cmd.c_str());
    REQUIRE(rc == 0);

    REQUIRE(fs::exists(log_path));
    REQUIRE(fs::exists(status_path));

    // training_log.json contract
    std::ifstream in_log(log_path);
    json L; in_log >> L;
    REQUIRE(L.contains("metadata"));
    REQUIRE(L.contains("snapshots"));
    auto& meta = L["metadata"];
    REQUIRE(meta["T_input"]      == 4);
    REQUIRE(meta["T_hidden"]     == 4);
    REQUIRE(meta["S_hidden"]     == 8);
    REQUIRE(meta["output_dim"]   == 10);
    REQUIRE(meta["cluster_size"] == 5);
    REQUIRE(meta["epochs"]       == 50);
    REQUIRE(meta["teacher_seed"] == 42);
    REQUIRE(meta["bfa_seed"]     == 43);

    // 50 epochs / interval 10 -> 5 snapshots
    REQUIRE(L["snapshots"].size() == 5);
    auto& last = L["snapshots"].back();
    REQUIRE(last["epoch"] == 50);
    REQUIRE(last.contains("loss"));
    REQUIRE(last.contains("lr"));

    auto& w = last["weights"];
    REQUIRE(w.contains("W1_T"));
    REQUIRE(w.contains("W2_T"));
    REQUIRE(w.contains("W1"));
    REQUIRE(w.contains("W2"));
    REQUIRE(w.contains("B_FA"));

    auto& s0 = last["samples"][0];
    REQUIRE(s0.contains("X"));
    REQUIRE(s0.contains("H_T"));
    REQUIRE(s0.contains("H_raw"));
    REQUIRE(s0.contains("Y_true"));
    REQUIRE(s0.contains("Y_pred"));
    REQUIRE(s0.contains("error"));
    REQUIRE(s0["X"].size()      == 4);     // T_input
    REQUIRE(s0["H_T"].size()    == 4);     // T_hidden
    REQUIRE(s0["H_raw"].size()  == 8);     // S_hidden
    REQUIRE(s0["Y_true"].size() == 10);
    REQUIRE(s0["Y_pred"].size() == 10);

    // training_status.json final state
    std::ifstream in_status(status_path);
    json S; in_status >> S;
    REQUIRE(S["running"]      == false);
    REQUIRE(S["completed_at"].is_string());
    REQUIRE(S["epoch"]        == 50);
    REQUIRE(S["total_epochs"] == 50);
}

TEST_CASE("benchmark CLI: --use-fhe smoke test (tiny)",
          "[benchmark][slow]") {
    auto dir = tmp_dir("fhe_smoke");
    auto log_path    = dir / "training_log.json";
    auto status_path = dir / "training_status.json";

    fs::path bench;
    if (const char* env = std::getenv("SSNS_BENCHMARK_BINARY")) {
        bench = env;
    } else {
        bench = "./ssns-benchmark";
        if (!fs::exists(bench)) bench = "build/ssns-benchmark";
    }
    REQUIRE(fs::exists(bench));

    // tiny config: 5 epochs, batch 4, S_hidden=4, output_dim=4
    // per-scalar CKKS keeps per-step cost at hundreds of ms; 5 epochs ~10s, fits any CI budget
    std::string cmd = bench.string()
        + " --t-input 2 --t-hidden 2 --s-input 2 --s-hidden 4"
        + " --output-dim 4 --cluster-size 2"
        + " --batch-size 4 --epochs 5 --snapshot-interval 5"
        + " --samples-to-log 1"
        + " --dz 0.10 --lr-max 0.01 --warmup-frac 0.05"
        + " --teacher-seed 42 --bfa-seed 43 --seed 2024"
        + " --use-fhe"
        + " --output-path " + log_path.string()
        + " --status-path " + status_path.string();

    const int rc = std::system(cmd.c_str());
    REQUIRE(rc == 0);
    REQUIRE(fs::exists(log_path));
    REQUIRE(fs::exists(status_path));

    std::ifstream in_status(status_path);
    json S; in_status >> S;
    REQUIRE(S["running"]      == false);
    REQUIRE(S["epoch"]        == 5);
    REQUIRE(S["total_epochs"] == 5);
}

TEST_CASE("benchmark CLI: rejects missing required flag",
          "[benchmark]") {
    fs::path bench;
    if (const char* env = std::getenv("SSNS_BENCHMARK_BINARY")) {
        bench = env;
    } else {
        bench = "./ssns-benchmark";
        if (!fs::exists(bench)) bench = "build/ssns-benchmark";
    }
    REQUIRE(fs::exists(bench));

    // missing --epochs; redirect stderr to keep test output clean, then assert non-zero rc
    std::string cmd = bench.string() + " --t-input 4 2>/dev/null";
    const int rc = std::system(cmd.c_str());
    REQUIRE(rc != 0);
}
