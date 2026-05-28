// io::StatusFile, lightweight ui_data/training_status.json polled by IDE every ~500ms
// mirrors Python's _write_starting_status (app.py) and write_status (scripts/benchmark_fhe_vs_plain.py)
#include <catch.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include <ssns/io/status_file.hpp>

using ssns::io::write_starting_status;
using ssns::io::write_progress;
using nlohmann::json;

// fresh temp dir under /tmp, removed if it already exists
static std::filesystem::path tmp_dir(const char* tag) {
    auto dir = std::filesystem::temp_directory_path()
             / ("ssns_status_test_" + std::string(tag));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

TEST_CASE("StatusFile: starting placeholder has running=true and completed_at=null",
          "[io][status]") {
    auto path = tmp_dir("starting") / "training_status.json";
    write_starting_status(path, /*total_epochs=*/1500);

    REQUIRE(std::filesystem::exists(path));
    std::ifstream in(path);
    json j; in >> j;

    REQUIRE(j["running"]      == true);
    REQUIRE(j["epoch"]        == 0);
    REQUIRE(j["total_epochs"] == 1500);
    REQUIRE(j["completed_at"].is_null());
    REQUIRE(j["loss"]         == 0.0);
    REQUIRE(j["lr"]           == 0.0);
    REQUIRE(j["elapsed_sec"]  == 0.0);
    REQUIRE(j["started_at"].is_string());
    REQUIRE(j["started_at"].get<std::string>().ends_with("Z"));
}

TEST_CASE("StatusFile: write_progress carries epoch / loss / running",
          "[io][status]") {
    auto path = tmp_dir("progress") / "training_status.json";
    auto t0 = std::chrono::system_clock::now();
    write_progress(path, /*epoch=*/350, /*total_epochs=*/1000,
                   /*loss=*/0.123, /*lr=*/0.008, /*started_at=*/t0,
                   /*running=*/true, /*completed_at=*/std::nullopt);

    std::ifstream in(path);
    json j; in >> j;
    REQUIRE(j["epoch"]        == 350);
    REQUIRE(j["total_epochs"] == 1000);
    REQUIRE(j["loss"]         == Approx(0.123));
    REQUIRE(j["lr"]           == Approx(0.008));
    REQUIRE(j["running"]      == true);
    REQUIRE(j["completed_at"].is_null());
    REQUIRE(j["started_at"].is_string());
    REQUIRE(j["elapsed_sec"].is_number());
}

TEST_CASE("StatusFile: write_progress with completed_at sets running=false flow",
          "[io][status]") {
    auto path = tmp_dir("complete") / "training_status.json";
    // elapsed_sec = (now-at-write-time - started_at), so build started_at 17s in the past and write now
    auto t1 = std::chrono::system_clock::now();
    auto t0 = t1 - std::chrono::seconds(17);
    write_progress(path, 1000, 1000, 0.001, 1e-9, t0, false, t1);

    std::ifstream in(path);
    json j; in >> j;
    REQUIRE(j["running"] == false);
    REQUIRE(j["completed_at"].is_string());
    REQUIRE(j["completed_at"].get<std::string>().ends_with("Z"));
    // elapsed_sec ~ 17.0, generous margin for any latency before write_progress timestamps internally
    REQUIRE(j["elapsed_sec"].get<double>() == Approx(17.0).margin(0.5));
}

TEST_CASE("StatusFile: atomic write - no .tmp leftover", "[io][status]") {
    auto dir = tmp_dir("atomic");
    auto path = dir / "training_status.json";
    write_starting_status(path, 100);

    bool any_tmp = false;
    for (auto& p : std::filesystem::directory_iterator(dir)) {
        if (p.path().extension() == ".tmp") { any_tmp = true; break; }
    }
    REQUIRE_FALSE(any_tmp);
    REQUIRE(std::filesystem::exists(path));
}

TEST_CASE("StatusFile: creates parent directories", "[io][status]") {
    auto root = std::filesystem::temp_directory_path() / "ssns_status_test_parents";
    std::filesystem::remove_all(root);
    auto path = root / "deep" / "nested" / "training_status.json";
    write_starting_status(path, 100);
    REQUIRE(std::filesystem::exists(path));
}
