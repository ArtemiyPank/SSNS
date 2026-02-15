// HTTP server hosting the SSNS Visual IDE — Phase 4 of the C++ port.
//
// Wraps cpp-httplib's Server.  Endpoints (paths, status codes, JSON bodies)
// match SSNS_mvp/app.py exactly so the frontend in ui/ runs unmodified.
//
// Endpoints:
//   GET  /                     -> ui/index.html
//   GET  /ui/<path>            -> static file under ui/
//   GET  /api/training_data    -> training_log.json (404 / 503 / 200)
//   GET  /api/training_status  -> training_status.json (404 / 503 / 200)
//   POST /api/run_training     -> spawn ssns-benchmark detached, 200 + pid
//   POST /api/manual_test      -> Teacher + Student forward on user X
//   POST /api/stress_test      -> batched stats over N random trials
//
// The class is intentionally light — each endpoint delegates to a free
// function in src/http/server.cpp so unit tests can grow against those
// helpers in a later phase if needed.
#ifndef SSNS_HTTP_SERVER_HPP
#define SSNS_HTTP_SERVER_HPP

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

// Forward-declare httplib::Server so we don't drag the 7k-line header into
// every TU that consumes this interface.
namespace httplib { class Server; }

namespace ssns::http {

struct ServerConfig {
    // Static UI assets root.  GET / serves <ui_dir>/index.html;
    // GET /ui/<path> serves <ui_dir>/<path>.
    std::filesystem::path ui_dir;
    // Where /api/training_data reads the training log JSON.
    std::filesystem::path training_data_path;
    // Where /api/training_status reads + /api/run_training writes the
    // running:true placeholder.
    std::filesystem::path training_status_path;
    // Path to the ssns-benchmark binary spawned by /api/run_training.
    // Tests substitute a no-op shell script here.
    std::filesystem::path benchmark_binary;
};

class Server {
public:
    explicit Server(ServerConfig cfg);
    ~Server();

    Server(const Server&)            = delete;
    Server& operator=(const Server&) = delete;

    // Blocks until stop() is invoked from another thread.
    void listen(const std::string& host, int port);

    // Convenience for tests: spin up listen() on a background thread.  The
    // thread is joined in stop() / dtor.
    void start_in_thread(const std::string& host, int port);

    // Idempotent.
    void stop();

private:
    ServerConfig cfg_;
    std::unique_ptr<httplib::Server> srv_;
    std::thread thread_;
    std::atomic<bool> running_{false};
};

}  // namespace ssns::http

#endif  // SSNS_HTTP_SERVER_HPP
