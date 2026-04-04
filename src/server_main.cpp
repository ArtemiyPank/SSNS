// ssns-server — HTTP server binary that hosts the Visual IDE backend.
// Mirrors `uvicorn app:app --host ... --port ...` from the Python reference.
//
// Usage:
//   ssns-server [--host 127.0.0.1] [--port 8765]
//               [--ui-dir ./ui] [--data ui_data/training_log.json]
//               [--status ui_data/training_status.json]
//               [--benchmark ./ssns-benchmark]
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include <ssns/http/server.hpp>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string host = "127.0.0.1";
    int port = 8765;
    fs::path ui_dir       = "ui";
    fs::path data_path    = "ui_data/training_log.json";
    fs::path status_path  = "ui_data/training_status.json";
    fs::path benchmark    = "";   // auto-resolved if left blank
    bool benchmark_explicit = false;
};

// Pick the most likely path to ssns-benchmark relative to CWD.  The
// canonical source layout puts it under build/, but if the user cd's into
// build/ and runs `./ssns-server` directly the binary is alongside.  We
// also fall back to PATH lookup via execvp's own search (handled by the
// HTTP layer), but report a clear error if no candidate exists at boot.
fs::path autodetect_benchmark() {
    const fs::path candidates[] = {
        "./build/ssns-benchmark",
        "./ssns-benchmark",
        "../build/ssns-benchmark",
    };
    for (const auto& c : candidates) {
        std::error_code ec;
        if (fs::exists(c, ec) && !ec) {
            return fs::weakly_canonical(c, ec);
        }
    }
    // Last-ditch: PATH lookup will be attempted at exec time.
    return "./ssns-benchmark";
}

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "ssns-server: " << msg << "\n";
    std::exit(2);
}

bool eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) die(std::string("missing value for ") + k);
            return argv[++i];
        };
        if      (eq(k, "--host"))      a.host = next();
        else if (eq(k, "--port"))      a.port = std::atoi(next());
        else if (eq(k, "--ui-dir"))    a.ui_dir = next();
        else if (eq(k, "--data"))      a.data_path = next();
        else if (eq(k, "--status"))    a.status_path = next();
        else if (eq(k, "--benchmark")) { a.benchmark = next(); a.benchmark_explicit = true; }
        else if (eq(k, "--help") || eq(k, "-h")) {
            std::cout
                << "ssns-server [--host H] [--port P] [--ui-dir D]\n"
                << "            [--data PATH] [--status PATH] [--benchmark PATH]\n";
            std::exit(0);
        }
        else die(std::string("unknown flag: ") + k);
    }
    return a;
}

// On startup, repair stale training_status.json: if it claims a run is
// active but no `ssns-benchmark` process is alive on this host, rewrite
// the file with running=false + stopped=true so the frontend doesn't
// permanently re-attach to a ghost run after the server is restarted.
void heal_stale_status(const fs::path& status_path) {
    if (!fs::exists(status_path)) return;
    nlohmann::json j;
    try {
        std::ifstream in(status_path);
        in >> j;
    } catch (...) { return; }
    if (!j.is_object() || !j.value("running", false)) return;

    // Cheap process check: scan /proc for any ssns-benchmark.  We don't
    // know the original pid (the file currently doesn't carry it); if any
    // benchmark is alive we leave the status alone.
    bool benchmark_alive = false;
    for (const auto& entry : fs::directory_iterator("/proc")) {
        if (!entry.is_directory()) continue;
        const auto& name = entry.path().filename().string();
        if (name.empty() || !std::isdigit(static_cast<unsigned char>(name[0]))) continue;
        std::ifstream comm(entry.path() / "comm");
        std::string s;
        if (comm && std::getline(comm, s) && s.find("ssns-benchmark") != std::string::npos) {
            benchmark_alive = true;
            break;
        }
    }
    if (benchmark_alive) return;

    const auto now_t = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now_t);
    std::tm tm_buf{}; gmtime_r(&tt, &tm_buf);
    char ts_buf[64];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);

    j["running"]      = false;
    j["stopped"]      = true;
    j["completed_at"] = std::string(ts_buf);
    try {
        std::ofstream out(status_path);
        out << j.dump();
        std::cout << "ssns-server: cleared stale running=true in "
                  << status_path << " (no live benchmark process)\n";
    } catch (...) { /* swallow */ }
}

}  // anon namespace

int main(int argc, char** argv) try {
    auto a = parse(argc, argv);
    if (!a.benchmark_explicit && a.benchmark.empty()) {
        a.benchmark = autodetect_benchmark();
    }
    {
        std::error_code ec;
        const bool exists = fs::exists(a.benchmark, ec) && !ec;
        if (!exists) {
            std::cerr << "ssns-server: WARNING benchmark binary not found at '"
                      << a.benchmark << "'.  /api/run_training will fail until "
                      << "you build ssns-benchmark or pass --benchmark <path>.\n";
        }
    }
    ssns::http::ServerConfig cfg{};
    cfg.ui_dir               = a.ui_dir;
    cfg.training_data_path   = a.data_path;
    cfg.training_status_path = a.status_path;
    cfg.benchmark_binary     = a.benchmark;

    heal_stale_status(cfg.training_status_path);

    ssns::http::Server srv(cfg);
    std::cout << "ssns-server listening on http://" << a.host << ":" << a.port
              << "  ui_dir=" << a.ui_dir
              << "  benchmark=" << a.benchmark << "\n";
    srv.listen(a.host, a.port);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "ssns-server: " << e.what() << "\n";
    return 2;
}
