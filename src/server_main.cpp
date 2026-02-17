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
#include <filesystem>
#include <iostream>
#include <string>

#include <ssns/http/server.hpp>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string host = "127.0.0.1";
    int port = 8765;
    fs::path ui_dir       = "ui";
    fs::path data_path    = "ui_data/training_log.json";
    fs::path status_path  = "ui_data/training_status.json";
    fs::path benchmark    = "./ssns-benchmark";
};

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
        else if (eq(k, "--benchmark")) a.benchmark = next();
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

}  // anon namespace

int main(int argc, char** argv) try {
    auto a = parse(argc, argv);
    ssns::http::ServerConfig cfg{};
    cfg.ui_dir               = a.ui_dir;
    cfg.training_data_path   = a.data_path;
    cfg.training_status_path = a.status_path;
    cfg.benchmark_binary     = a.benchmark;

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
