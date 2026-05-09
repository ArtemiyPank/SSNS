// http server for ssns visual ide
//
// wraps cpp-httplib server
// endpoints status codes and json bodies match python ref so ui works as is
//
// endpoints
//   GET  /                     -> ui/index.html
//   GET  /ui/<path>            -> static file under ui
//   GET  /api/training_data    -> training_log json (404 / 503 / 200)
//   GET  /api/training_status  -> training_status json (404 / 503 / 200)
//   POST /api/run_training     -> spawn ssns-benchmark detached 200 + pid
//   POST /api/stop_training    -> sigterm running benchmark
//   POST /api/manual_test      -> teacher + student forward on user X
//   POST /api/stress_test      -> batched stats over N random trials
//
// thin class each endpoint forwards to a free fn in src/http/server.cpp
#ifndef SSNS_HTTP_SERVER_HPP
#define SSNS_HTTP_SERVER_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <thread>

// fwd decl httplib::Server so we dont pull the 7k line header everywhere
namespace httplib { class Server; }

namespace ssns::http {

struct ServerConfig {
    // static ui assets root, GET / serves <ui_dir>/index.html and /ui/<path>
    std::filesystem::path ui_dir;
    // path read by /api/training_data
    std::filesystem::path training_data_path;
    // path read by /api/training_status and written by /api/run_training placeholder
    std::filesystem::path training_status_path;
    // path to ssns-benchmark binary spawned by /api/run_training
    std::filesystem::path benchmark_binary;
};

class Server {
public:
    explicit Server(ServerConfig cfg);
    ~Server();

    Server(const Server&)            = delete;
    Server& operator=(const Server&) = delete;

    // blocks till stop() called from another thread
    void listen(const std::string& host, int port);

    // test helper runs listen on bg thread
    // joined in stop() and dtor
    void start_in_thread(const std::string& host, int port);

    // idempotent
    void stop();

private:
    ServerConfig cfg_;
    std::unique_ptr<httplib::Server> srv_;
    std::thread thread_;
};

}  // namespace ssns::http

#endif  // SSNS_HTTP_SERVER_HPP
