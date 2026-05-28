// HTTP server integration tests; phase 4 of the SSNS C++ port
//
// each test boots a Server on a unique port (18800 + idx), drives it via httplib::Client
// heavy state (training_log.json, training_status.json) lives in per-test temp dirs
//
// /api/run_training never invokes the real ssns-benchmark; a shell stub touches output files
// and exits (fast, still exercises fork + execvp + detach)
#include <catch.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <sys/wait.h>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <ssns/http/server.hpp>
#include <ssns/io/status_file.hpp>

namespace fs = std::filesystem;
using nlohmann::json;
using ssns::http::Server;
using ssns::http::ServerConfig;

namespace {

// per-test scratch dir
fs::path tmp_dir(const std::string& tag) {
    auto dir = fs::temp_directory_path() / ("ssns_http_test_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// minimal ui/ dir with just one index.html
fs::path make_ui_dir(const std::string& tag, const std::string& body) {
    auto dir = tmp_dir(tag + "_ui");
    std::ofstream(dir / "index.html") << body;
    return dir;
}

// write JSON at `path` so server-side reads succeed
void write_json(const fs::path& path, const json& j) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << j.dump();
}

// wait until server.listen() accepts connections
// cpp-httplib has no public is_running() that returns true BEFORE listen blocks
// so we poll a known endpoint for up to ~2s
bool wait_until_ready(const std::string& host, int port, int timeout_ms = 2000) {
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(0, 100 * 1000);   // 100 ms
        auto r = cli.Get("/__ping");                 // any path; 404 is fine
        if (r) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

// synthetic training_log.json with one snapshot
// enough for /api/manual_test + /api/stress_test to rebuild a Teacher and run a Student forward
// tiny dims keep the test fast
json make_training_log() {
    constexpr int T_input    = 4;
    constexpr int T_hidden   = 4;
    constexpr int S_input    = 4;
    constexpr int S_hidden   = 8;
    constexpr int output_dim = 10;
    constexpr int cluster    = 5;

    // small random W1/W2; values needn't make bits agree
    // we just need the endpoint to reduce them to bits without crashing
    std::mt19937_64 rng(123);
    std::normal_distribution<double> nd(0.0, 0.1);

    auto rand_mat = [&](int rows, int cols) {
        json m = json::array();
        for (int r = 0; r < rows; ++r) {
            json row = json::array();
            for (int c = 0; c < cols; ++c) row.push_back(nd(rng));
            m.push_back(row);
        }
        return m;
    };

    json log;
    log["metadata"] = {
        {"T_input", T_input}, {"T_hidden", T_hidden},
        {"S_input", S_input}, {"S_hidden", S_hidden},
        {"output_dim", output_dim}, {"cluster_size", cluster},
        {"batch_size", 8}, {"epochs", 50},
        {"dz", 0.10}, {"lr_max", 0.01}, {"warmup_frac", 0.05},
        {"snapshot_interval", 10}, {"samples_logged", 2},
        {"teacher_seed", 42}, {"bfa_seed", 43},
    };

    json snap;
    snap["epoch"] = 50;
    snap["loss"]  = 0.123;
    snap["lr"]    = 1e-9;
    snap["weights"] = {
        {"W1_T", rand_mat(T_input,  T_hidden)},
        {"W2_T", rand_mat(T_hidden, output_dim)},
        {"W1",   rand_mat(S_input,  S_hidden)},
        {"W2",   rand_mat(S_hidden, output_dim)},
        {"B_FA", rand_mat(output_dim, S_hidden)},
    };
    snap["samples"] = json::array();
    log["snapshots"] = json::array({snap});
    return log;
}

}  // anon namespace

// test 1: GET / returns ui/index.html

TEST_CASE("HTTP: GET / serves ui/index.html", "[http][routes]") {
    constexpr int port = 18801;
    auto ui = make_ui_dir("root", "<!doctype html><title>HELLO_SSNS</title>");

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = tmp_dir("root_data") / "training_log.json";
    cfg.training_status_path= tmp_dir("root_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Get("/");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    REQUIRE(r->body.find("HELLO_SSNS") != std::string::npos);

    srv.stop();
}

TEST_CASE("HTTP: GET / 404 when index.html missing", "[http][routes]") {
    constexpr int port = 18802;
    auto ui = tmp_dir("missing_ui");   // empty, no index.html
    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = tmp_dir("missing_data") / "training_log.json";
    cfg.training_status_path= tmp_dir("missing_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Get("/");
    REQUIRE(r);
    REQUIRE(r->status == 404);
    srv.stop();
}

// test 2: GET /ui/<file> static serving

TEST_CASE("HTTP: GET /ui/<file> serves static asset with correct MIME",
          "[http][static]") {
    constexpr int port = 18803;
    auto ui = make_ui_dir("static", "");
    std::ofstream(ui / "app.css") << "body{}\n";

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = tmp_dir("static_data") / "training_log.json";
    cfg.training_status_path= tmp_dir("static_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Get("/ui/app.css");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    REQUIRE(r->body == "body{}\n");
    REQUIRE(r->get_header_value("Content-Type").find("text/css") != std::string::npos);

    srv.stop();
}

// test 3: GET /api/training_data - 404 when file missing, 200 otherwise

TEST_CASE("HTTP: GET /api/training_data - 404 when missing", "[http][api]") {
    constexpr int port = 18804;
    auto ui = make_ui_dir("td404", "<html></html>");

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = tmp_dir("td404_data") / "training_log.json";
    cfg.training_status_path= tmp_dir("td404_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Get("/api/training_data");
    REQUIRE(r);
    REQUIRE(r->status == 404);
    srv.stop();
}

TEST_CASE("HTTP: GET /api/training_data - 200 returns log", "[http][api]") {
    constexpr int port = 18805;
    auto ui = make_ui_dir("td200", "<html></html>");
    auto data_path = tmp_dir("td200_data") / "training_log.json";
    write_json(data_path, make_training_log());

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = data_path;
    cfg.training_status_path= tmp_dir("td200_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Get("/api/training_data");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    auto j = json::parse(r->body);
    REQUIRE(j["metadata"]["epochs"] == 50);
    REQUIRE(j["snapshots"].size()   == 1);
    srv.stop();
}

// test 4: GET /api/training_status, 404 when missing

TEST_CASE("HTTP: GET /api/training_status - 404 when missing", "[http][api]") {
    constexpr int port = 18806;
    auto ui = make_ui_dir("ts404", "<html></html>");

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = tmp_dir("ts404_data") / "training_log.json";
    cfg.training_status_path= tmp_dir("ts404_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Get("/api/training_status");
    REQUIRE(r);
    REQUIRE(r->status == 404);
    srv.stop();
}

TEST_CASE("HTTP: GET /api/training_status - 200 returns status", "[http][api]") {
    constexpr int port = 18807;
    auto ui = make_ui_dir("ts200", "<html></html>");
    auto status_path = tmp_dir("ts200_status") / "training_status.json";
    ssns::io::write_starting_status(status_path, /*total_epochs=*/100);

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = tmp_dir("ts200_data") / "training_log.json";
    cfg.training_status_path= status_path;
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Get("/api/training_status");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    auto j = json::parse(r->body);
    REQUIRE(j["running"]      == true);
    REQUIRE(j["total_epochs"] == 100);
    srv.stop();
}

// test 5: POST /api/run_training validates body, then spawns the subprocess

TEST_CASE("HTTP: POST /api/run_training - 422 on missing field", "[http][api]") {
    constexpr int port = 18808;
    auto ui = make_ui_dir("rt422", "<html></html>");

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = tmp_dir("rt422_data") / "training_log.json";
    cfg.training_status_path= tmp_dir("rt422_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    json body = {{"T_input", 64}};   // missing every other required field
    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Post("/api/run_training", body.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 422);
    auto j = json::parse(r->body);
    REQUIRE(j.contains("detail"));
    srv.stop();
}

TEST_CASE("HTTP: POST /api/run_training - spawns subprocess + writes placeholder",
          "[http][api]") {
    constexpr int port = 18809;
    auto ui      = make_ui_dir("rtok", "<html></html>");
    auto td_dir  = tmp_dir("rtok_data");
    auto ts_dir  = tmp_dir("rtok_status");
    auto data_path   = td_dir / "training_log.json";
    auto status_path = ts_dir / "training_status.json";

    // stub benchmark binary writes a sentinel at $1; server passes it via execvp argv[0]
    // we don't care about the rest of args as long as the subprocess ran
    auto stub = tmp_dir("rtok_stub") / "stub.sh";
    auto sentinel = stub.parent_path() / "ran.txt";
    {
        std::ofstream s(stub);
        s << "#!/bin/sh\n"
          << "echo ok > '" << sentinel.string() << "'\n"
          << "exit 0\n";
    }
    fs::permissions(stub, fs::perms::owner_all | fs::perms::group_read
                          | fs::perms::group_exec | fs::perms::others_read
                          | fs::perms::others_exec);

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = data_path;
    cfg.training_status_path= status_path;
    cfg.benchmark_binary    = stub;
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    json body = {
        {"T_input", 4}, {"T_hidden", 4},
        {"S_input", 4}, {"S_hidden", 8},
        {"output_dim", 10}, {"cluster_size", 5},
        {"batch_size", 8}, {"epochs", 50},
        {"dz", 0.10}, {"lr_max", 0.01}, {"warmup_frac", 0.05},
    };
    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Post("/api/run_training", body.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    auto j = json::parse(r->body);
    REQUIRE(j["status"] == "started");
    REQUIRE(j.contains("pid"));
    REQUIRE(j["pid"].is_number_integer());
    REQUIRE(j["params"]["T_input"] == 4);
    REQUIRE(j["cmd"].is_array());
    REQUIRE(j["cmd"][0] == stub.string());

    // placeholder status file must exist after server returns
    REQUIRE(fs::exists(status_path));
    {
        std::ifstream in(status_path);
        json s; in >> s;
        REQUIRE(s["running"] == true);
        REQUIRE(s["total_epochs"] == 50);
    }

    // subprocess should have run and dropped its sentinel; poll up to 2s
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(sentinel)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(fs::exists(sentinel));

    srv.stop();
}

// /api/stop_training: input validation + happy-path SIGTERM

TEST_CASE("HTTP: POST /api/stop_training - 422 on missing pid", "[http][api]") {
    constexpr int port = 18820;
    auto ui = make_ui_dir("stop422", "<html></html>");
    ServerConfig cfg{};
    cfg.ui_dir               = ui;
    cfg.training_data_path   = tmp_dir("stop422_d") / "training_log.json";
    cfg.training_status_path = tmp_dir("stop422_s") / "training_status.json";
    cfg.benchmark_binary     = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    httplib::Client cli("127.0.0.1", port);
    auto r1 = cli.Post("/api/stop_training", "{}", "application/json");
    REQUIRE(r1);
    REQUIRE(r1->status == 422);

    auto r2 = cli.Post("/api/stop_training",
                       json{{"pid", 1}}.dump(), "application/json");
    REQUIRE(r2);
    REQUIRE(r2->status == 422);  // pid must be > 1

    auto r3 = cli.Post("/api/stop_training",
                       json{{"pid", 999999999}}.dump(), "application/json");
    REQUIRE(r3);
    REQUIRE(r3->status == 404);  // not a benchmark process

    srv.stop();
}

TEST_CASE("HTTP: POST /api/stop_training - kills a child and writes stopped status",
          "[http][api]") {
    constexpr int port = 18821;
    auto ui = make_ui_dir("stopok", "<html></html>");
    auto status_path = tmp_dir("stopok_status") / "training_status.json";

    // tiny stub script renamed to look like ssns-benchmark
    // stop handler validates pid via /proc/<pid>/comm or /cmdline matching "ssns-benchmark"
    // so the filename must include that
    auto stub_dir = tmp_dir("stopok_stub");
    auto stub     = stub_dir / "ssns-benchmark";  // name matters
    {
        std::ofstream s(stub);
        s << "#!/bin/sh\n"
          << "trap 'exit 0' TERM\n"
          << "while true; do sleep 0.1; done\n";
    }
    fs::permissions(stub, fs::perms::owner_all | fs::perms::group_read
                          | fs::perms::group_exec | fs::perms::others_read
                          | fs::perms::others_exec);

    ServerConfig cfg{};
    cfg.ui_dir               = ui;
    cfg.training_data_path   = tmp_dir("stopok_d") / "training_log.json";
    cfg.training_status_path = status_path;
    cfg.benchmark_binary     = stub;
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    // spawn via /api/run_training so the stub becomes our child
    json body = {
        {"T_input", 4}, {"T_hidden", 4}, {"S_input", 4}, {"S_hidden", 8},
        {"output_dim", 10}, {"cluster_size", 5},
        {"batch_size", 8}, {"epochs", 100},
        {"dz", 0.1}, {"lr_max", 0.01}, {"warmup_frac", 0.05},
        {"snapshot_interval", 10}, {"samples_to_log", 2},
        {"teacher_seed", 42}, {"bfa_seed", 43}, {"seed", 2024},
        {"use_fhe", false},
    };
    httplib::Client cli("127.0.0.1", port);
    auto rr = cli.Post("/api/run_training", body.dump(), "application/json");
    REQUIRE(rr);
    REQUIRE(rr->status == 200);
    auto rrj = json::parse(rr->body);
    REQUIRE(rrj.contains("pid"));
    const long long pid = rrj["pid"].get<long long>();
    REQUIRE(pid > 1);

    // wait briefly for the child to settle into its sleep loop so /proc/<pid> is fully populated (comm + cmdline)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto rs = cli.Post("/api/stop_training",
                       json{{"pid", pid}}.dump(), "application/json");
    REQUIRE(rs);
    REQUIRE(rs->status == 200);
    auto rsj = json::parse(rs->body);
    REQUIRE(rsj["status"]  == "stopped");
    REQUIRE(rsj["pid"]     == pid);
    REQUIRE(rsj["signal"]  == "SIGTERM");

    // final status JSON must show running:false and stopped:true
    std::ifstream in(status_path);
    REQUIRE(in.good());
    json final_j; in >> final_j;
    REQUIRE(final_j["running"] == false);
    REQUIRE(final_j["stopped"] == true);
    REQUIRE(final_j["completed_at"].is_string());

    // reap the child so it does not stay as a zombie
    waitpid(static_cast<pid_t>(pid), nullptr, WNOHANG);
    srv.stop();
}

// test 6: POST /api/manual_test happy path

TEST_CASE("HTTP: POST /api/manual_test - happy path returns shape-correct payload",
          "[http][api]") {
    constexpr int port = 18810;
    auto ui      = make_ui_dir("mt", "<html></html>");
    auto data_path = tmp_dir("mt_data") / "training_log.json";
    write_json(data_path, make_training_log());

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = data_path;
    cfg.training_status_path= tmp_dir("mt_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    json body = {{"X", {0.1, -0.2, 0.3, 0.4}}};   // S_input == 4
    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Post("/api/manual_test", body.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    auto j = json::parse(r->body);
    REQUIRE(j["Y_T"].size() == 10);
    REQUIRE(j["Y_S"].size() == 10);
    REQUIRE(j.contains("bits_T"));
    REQUIRE(j.contains("bits_S"));
    REQUIRE(j.contains("idx_T"));
    REQUIRE(j.contains("idx_S"));
    REQUIRE(j.contains("shared_indices"));
    REQUIRE(j.contains("mismatches"));
    REQUIRE(j.contains("n_confident_T"));
    REQUIRE(j.contains("n_confident_S"));
    REQUIRE(j.contains("match"));
    REQUIRE(j["epoch_used"] == 50);
    srv.stop();
}

TEST_CASE("HTTP: POST /api/manual_test - 422 on wrong X length",
          "[http][api]") {
    constexpr int port = 18811;
    auto ui = make_ui_dir("mt422", "<html></html>");
    auto data_path = tmp_dir("mt422_data") / "training_log.json";
    write_json(data_path, make_training_log());

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = data_path;
    cfg.training_status_path= tmp_dir("mt422_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    json body = {{"X", {0.1, 0.2}}};   // length 2, S_input is 4
    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Post("/api/manual_test", body.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 422);
    srv.stop();
}

// test 7: POST /api/stress_test happy path

TEST_CASE("HTTP: POST /api/stress_test - returns aggregate stats",
          "[http][api]") {
    constexpr int port = 18812;
    auto ui      = make_ui_dir("st", "<html></html>");
    auto data_path = tmp_dir("st_data") / "training_log.json";
    write_json(data_path, make_training_log());

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = data_path;
    cfg.training_status_path= tmp_dir("st_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    json body = {{"n_trials", 5}, {"seed", 7}};
    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Post("/api/stress_test", body.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    auto j = json::parse(r->body);
    REQUIRE(j["n_trials"]       == 5);
    REQUIRE(j["epoch_used"]     == 50);
    REQUIRE(j["total_clusters"] == 2);          // 10 / 5
    REQUIRE(j["cluster_size"]   == 5);
    REQUIRE(j.contains("teacher_confident"));
    REQUIRE(j.contains("student_confident"));
    REQUIRE(j.contains("shared"));
    REQUIRE(j.contains("mismatches"));
    REQUIRE(j.contains("match_rate_pct"));
    REQUIRE(j.contains("histogram_shared"));
    REQUIRE(j["histogram_shared"]["bins"].is_array());
    REQUIRE(j["histogram_shared"]["counts"].is_array());
    srv.stop();
}

TEST_CASE("HTTP: POST /api/stress_test - 422 on out-of-range n_trials",
          "[http][api]") {
    constexpr int port = 18813;
    auto ui = make_ui_dir("st422", "<html></html>");
    auto data_path = tmp_dir("st422_data") / "training_log.json";
    write_json(data_path, make_training_log());

    ServerConfig cfg{};
    cfg.ui_dir              = ui;
    cfg.training_data_path  = data_path;
    cfg.training_status_path= tmp_dir("st422_status") / "training_status.json";
    cfg.benchmark_binary    = "/bin/true";
    Server srv(cfg);
    srv.start_in_thread("127.0.0.1", port);
    REQUIRE(wait_until_ready("127.0.0.1", port));

    json body = {{"n_trials", 0}};
    httplib::Client cli("127.0.0.1", port);
    auto r = cli.Post("/api/stress_test", body.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 422);
    srv.stop();
}
