// HTTP server implementation.  Each endpoint is a small free function
// registered against an httplib::Server instance owned by the wrapper.
//
// Subprocess-spawn for /api/run_training uses POSIX fork()+execvp() so the
// child can fully detach (setsid + close stdio + replace image), matching
// what subprocess.Popen with start_new_session=True does in the Python
// reference.
#include <ssns/http/server.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <ssns/io/status_file.hpp>
#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/activations.hpp>
#include <ssns/nn/teacher.hpp>
#include <ssns/protocol/bit_extract.hpp>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ssns::http {

namespace {

using nlohmann::json;
namespace fs = std::filesystem;

// ----- shared helpers ------------------------------------------------------

void send_error(httplib::Response& res, int status, const std::string& detail) {
    json body = {{"detail", detail}};
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

// JSON read with retry (3 x 50ms backoff) on parse error.  Mirrors
// _read_json_with_retry in app.py.
json read_json_with_retry(const fs::path& path,
                          int attempts = 3,
                          int backoff_ms = 50) {
    json last;
    bool last_ok = false;
    std::string last_err;
    for (int i = 0; i < attempts; ++i) {
        std::ifstream in(path);
        if (!in) {
            last_err = "cannot open file";
            break;
        }
        try {
            in >> last;
            last_ok = true;
            break;
        } catch (const json::parse_error& e) {
            last_err = e.what();
            if (i + 1 < attempts) {
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            }
        }
    }
    if (!last_ok) {
        throw std::runtime_error(last_err);
    }
    return last;
}

// ----- static file serving -------------------------------------------------

std::string mime_for(const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
    if (ext == ".css")  return "text/css; charset=utf-8";
    if (ext == ".js")   return "application/javascript; charset=utf-8";
    if (ext == ".json") return "application/json; charset=utf-8";
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif")  return "image/gif";
    if (ext == ".svg")  return "image/svg+xml";
    if (ext == ".ico")  return "image/x-icon";
    if (ext == ".txt")  return "text/plain; charset=utf-8";
    return "application/octet-stream";
}

bool read_file(const fs::path& p, std::string& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

// Reject ".." traversal — only allow simple relative descent under ui_dir.
bool is_safe_subpath(const std::string& sub) {
    if (sub.empty()) return false;
    if (sub.find("..") != std::string::npos) return false;
    if (sub.front() == '/') return false;
    return true;
}

// ----- /api/run_training subprocess spawn ----------------------------------

// Fields required in the POST body.  Matches TrainingParams in app.py.
struct TrainingParams {
    long T_input, T_hidden, S_input, S_hidden;
    long output_dim, cluster_size, batch_size, epochs;
    double dz, lr_max, warmup_frac;
    long samples_to_log = 20;
    long snapshot_count = 10;
    bool use_fhe = false;
};

// Returns nullopt + error string in `err` on missing/invalid field.
bool parse_training_params(const json& body, TrainingParams& out, std::string& err) {
    auto need_int = [&](const char* name, long& dst) -> bool {
        if (!body.contains(name)) {
            err = std::string("missing field: ") + name;
            return false;
        }
        const auto& v = body.at(name);
        if (!v.is_number_integer() && !v.is_number_unsigned()) {
            err = std::string("field must be int: ") + name;
            return false;
        }
        dst = v.get<long>();
        return true;
    };
    auto need_double = [&](const char* name, double& dst) -> bool {
        if (!body.contains(name)) {
            err = std::string("missing field: ") + name;
            return false;
        }
        const auto& v = body.at(name);
        if (!v.is_number()) {
            err = std::string("field must be number: ") + name;
            return false;
        }
        dst = v.get<double>();
        return true;
    };

    if (!body.is_object()) { err = "request body must be a JSON object"; return false; }
    if (!need_int("T_input",      out.T_input))      return false;
    if (!need_int("T_hidden",     out.T_hidden))     return false;
    if (!need_int("S_input",      out.S_input))      return false;
    if (!need_int("S_hidden",     out.S_hidden))     return false;
    if (!need_int("output_dim",   out.output_dim))   return false;
    if (!need_int("cluster_size", out.cluster_size)) return false;
    if (!need_int("batch_size",   out.batch_size))   return false;
    if (!need_int("epochs",       out.epochs))       return false;
    if (!need_double("dz",          out.dz))          return false;
    if (!need_double("lr_max",      out.lr_max))      return false;
    if (!need_double("warmup_frac", out.warmup_frac)) return false;

    if (body.contains("samples_to_log")) {
        if (!body["samples_to_log"].is_number_integer()
            && !body["samples_to_log"].is_number_unsigned()) {
            err = "samples_to_log must be int"; return false;
        }
        out.samples_to_log = body["samples_to_log"].get<long>();
        if (out.samples_to_log < 1 || out.samples_to_log > 1024) {
            err = "samples_to_log out of range [1,1024]"; return false;
        }
    }
    if (body.contains("snapshot_count")) {
        if (!body["snapshot_count"].is_number_integer()
            && !body["snapshot_count"].is_number_unsigned()) {
            err = "snapshot_count must be int"; return false;
        }
        out.snapshot_count = body["snapshot_count"].get<long>();
        if (out.snapshot_count < 2 || out.snapshot_count > 200) {
            err = "snapshot_count out of range [2,200]"; return false;
        }
    }
    if (body.contains("use_fhe")) {
        if (!body["use_fhe"].is_boolean()) {
            err = "use_fhe must be bool"; return false;
        }
        out.use_fhe = body["use_fhe"].get<bool>();
    }
    return true;
}

// Build argv list mirroring _PARAM_TO_CLI in app.py.  First entry is the
// benchmark binary path itself (becomes argv[0]).
std::vector<std::string> build_cmd_argv(const fs::path& benchmark,
                                        const TrainingParams& p) {
    std::vector<std::string> a;
    a.push_back(benchmark.string());
    auto push = [&](const char* flag, const std::string& v) {
        a.emplace_back(flag); a.emplace_back(v);
    };
    push("--t-input",       std::to_string(p.T_input));
    push("--t-hidden",      std::to_string(p.T_hidden));
    push("--s-input",       std::to_string(p.S_input));
    push("--s-hidden",      std::to_string(p.S_hidden));
    push("--output-dim",    std::to_string(p.output_dim));
    push("--cluster-size",  std::to_string(p.cluster_size));
    push("--batch-size",    std::to_string(p.batch_size));
    push("--epochs",        std::to_string(p.epochs));
    {
        std::ostringstream ss; ss << p.dz;          push("--dz",          ss.str());
    }
    {
        std::ostringstream ss; ss << p.lr_max;      push("--lr-max",      ss.str());
    }
    {
        std::ostringstream ss; ss << p.warmup_frac; push("--warmup-frac", ss.str());
    }
    push("--samples-to-log", std::to_string(p.samples_to_log));

    long snap_count = std::max<long>(1L, p.snapshot_count);
    long snap_interval = std::max<long>(1L, p.epochs / snap_count);
    push("--snapshot-interval", std::to_string(snap_interval));

    a.emplace_back("--skip-fhe-bench");
    if (p.use_fhe) a.emplace_back("--use-fhe");
    return a;
}

#if defined(__unix__) || defined(__APPLE__)

// fork+execvp; child detaches via setsid and redirects stdio to /dev/null
// before exec.  Returns child pid on success, -1 on fork failure.
pid_t spawn_detached(const std::vector<std::string>& argv) {
    if (argv.empty()) return -1;

    // Avoid zombies: install SIG_IGN for SIGCHLD on first call.  Children
    // will be auto-reaped by the kernel.  Idempotent + harmless.
    static bool sigchld_installed = false;
    if (!sigchld_installed) {
        struct sigaction sa{};
        sa.sa_handler = SIG_IGN;
        sa.sa_flags   = SA_NOCLDWAIT;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGCHLD, &sa, nullptr);
        sigchld_installed = true;
    }

    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        // Child.
        setsid();
        int dn = ::open("/dev/null", O_RDWR);
        if (dn >= 0) {
            dup2(dn, STDIN_FILENO);
            dup2(dn, STDOUT_FILENO);
            dup2(dn, STDERR_FILENO);
            if (dn > STDERR_FILENO) ::close(dn);
        }
        // Build argv as char* array.
        std::vector<char*> raw;
        raw.reserve(argv.size() + 1);
        for (auto& s : argv) {
            raw.push_back(const_cast<char*>(s.c_str()));
        }
        raw.push_back(nullptr);
        execvp(raw[0], raw.data());
        _exit(127);
    }
    return pid;
}

#else

pid_t spawn_detached(const std::vector<std::string>&) {
    return -1;
}

#endif

// ----- /api/manual_test + /api/stress_test helpers -------------------------

// Convert nested JSON 2-D array to ssns::linalg::Matrix.
ssns::linalg::Matrix matrix_from_json(const json& m) {
    if (!m.is_array() || m.empty() || !m.front().is_array()) {
        throw std::runtime_error("expected 2-D JSON array");
    }
    const std::size_t rows = m.size();
    const std::size_t cols = m.front().size();
    ssns::linalg::Matrix M(rows, cols);
    for (std::size_t r = 0; r < rows; ++r) {
        const auto& row = m[r];
        if (!row.is_array() || row.size() != cols) {
            throw std::runtime_error("ragged 2-D array");
        }
        for (std::size_t c = 0; c < cols; ++c) {
            M(r, c) = row[c].get<double>();
        }
    }
    return M;
}

struct LoadedPair {
    json metadata;
    json last_snapshot;
    ssns::nn::Teacher teacher;
    ssns::linalg::Matrix W1_S;
    ssns::linalg::Matrix W2_S;
};

// Mirror of app.py::_load_trained_pair.  Throws an HTTP-friendly status code
// via std::pair<int,std::string> packaged into a runtime_error tagged with a
// numeric prefix; we unpack via parse_status().  Keeps the call sites tiny.
struct HTTPException : public std::runtime_error {
    int status;
    HTTPException(int s, const std::string& m) : std::runtime_error(m), status(s) {}
};

LoadedPair load_trained_pair(const fs::path& data_path) {
    if (!fs::exists(data_path)) {
        throw HTTPException(404,
            "Training log not found at " + data_path.string()
            + ". Run training first.");
    }
    json data;
    try {
        data = read_json_with_retry(data_path);
    } catch (const std::exception&) {
        throw HTTPException(503,
            "Training log is currently being written. Retry shortly.");
    }
    if (!data.contains("metadata") || !data.contains("snapshots")) {
        throw HTTPException(409, "Training log missing 'metadata' or 'snapshots'.");
    }
    const auto& meta = data["metadata"];
    const auto& snaps = data["snapshots"];
    if (!snaps.is_array() || snaps.empty()) {
        throw HTTPException(409,
            "Training log has no snapshots yet — wait for training to "
            "produce at least one snapshot.");
    }
    if (!meta.contains("S_input") || meta["S_input"].is_null()) {
        throw HTTPException(409, "Training log metadata missing 'S_input'.");
    }
    if (!meta.contains("teacher_seed") || meta["teacher_seed"].is_null()) {
        throw HTTPException(409,
            "Training log lacks 'teacher_seed' metadata — re-run training "
            "with the latest benchmark.");
    }
    const std::size_t T_input    = meta.at("T_input").get<std::size_t>();
    const std::size_t T_hidden   = meta.at("T_hidden").get<std::size_t>();
    const std::size_t output_dim = meta.at("output_dim").get<std::size_t>();
    const std::uint64_t teacher_seed = meta.at("teacher_seed").get<std::uint64_t>();

    ssns::nn::Teacher teacher(T_input, T_hidden, output_dim, teacher_seed);
    const auto& last = snaps.back();
    auto W1_S = matrix_from_json(last.at("weights").at("W1"));
    auto W2_S = matrix_from_json(last.at("weights").at("W2"));
    return LoadedPair{meta, last, std::move(teacher), std::move(W1_S), std::move(W2_S)};
}

// 1-D summary stats — mean / std (population) / min / max / median.
json summary(const std::vector<double>& v) {
    if (v.empty()) {
        return json{{"mean",0.0},{"std",0.0},{"min",0.0},{"max",0.0},{"median",0.0}};
    }
    double mn = v[0], mx = v[0], sum = 0.0;
    for (double x : v) { mn = std::min(mn, x); mx = std::max(mx, x); sum += x; }
    const double mean = sum / static_cast<double>(v.size());
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= static_cast<double>(v.size());     // population (unbiased=False)
    auto sorted = v;
    std::sort(sorted.begin(), sorted.end());
    double median;
    const std::size_t n = sorted.size();
    if (n % 2 == 1) median = sorted[n / 2];
    // PyTorch's tensor.median picks the lower middle for even n; mirror that.
    else            median = sorted[n / 2 - 1];
    return json{
        {"mean",   mean},
        {"std",    std::sqrt(var)},
        {"min",    mn},
        {"max",    mx},
        {"median", median},
    };
}

// Fixed-bin histogram to match _hist_counts in app.py.
json hist_counts(const std::vector<long>& values, int n_bins,
                 long lo, long hi) {
    if (values.empty()) {
        return json{{"bins", {0.0, 1.0}}, {"counts", {0}}};
    }
    if (hi == lo) hi = lo + 1;
    const double width = static_cast<double>(hi - lo) / n_bins;
    std::vector<double> edges(n_bins + 1);
    for (int i = 0; i <= n_bins; ++i) {
        edges[i] = lo + i * width;
    }
    std::vector<long> counts(n_bins, 0);
    for (long v : values) {
        long idx = (width > 0)
            ? static_cast<long>((v - lo) / width)
            : 0L;
        if (idx >= n_bins) idx = n_bins - 1;
        if (idx < 0)       idx = 0;
        counts[idx]++;
    }
    return json{{"bins", edges}, {"counts", counts}};
}

// Apply Student forward Y = sigmoid(ReLU(X @ W1) @ W2) row-by-row to a
// batched X with shape [n, S_input].  Uses the C++ matrix kernels.
ssns::linalg::Matrix student_forward_sigmoid(
    const ssns::linalg::Matrix& X,
    const ssns::linalg::Matrix& W1,
    const ssns::linalg::Matrix& W2)
{
    auto pre   = ssns::linalg::matmul(X, W1);
    auto post  = ssns::nn::relu(pre);
    auto raw   = ssns::linalg::matmul(post, W2);
    return ssns::nn::sigmoid(raw);
}

ssns::linalg::Matrix teacher_forward_sigmoid(
    const ssns::nn::Teacher& teacher,
    const ssns::linalg::Matrix& X)
{
    auto raw = teacher.forward(X);
    return ssns::nn::sigmoid(raw);
}

// ----- endpoint handlers ---------------------------------------------------

void handle_root(const ServerConfig& cfg, const httplib::Request&, httplib::Response& res) {
    auto index = cfg.ui_dir / "index.html";
    if (!fs::exists(index)) {
        send_error(res, 404,
            "ui/index.html not found at " + index.string()
            + ". Open one of the static files at /ui/<filename>.");
        return;
    }
    std::string body;
    if (!read_file(index, body)) {
        send_error(res, 500, "failed to read index.html");
        return;
    }
    res.status = 200;
    res.set_content(body, "text/html; charset=utf-8");
}

void handle_static(const ServerConfig& cfg, const httplib::Request& req,
                   httplib::Response& res) {
    // matches.size() == 2: full match + capture group
    if (req.matches.size() < 2) {
        send_error(res, 400, "missing static path");
        return;
    }
    const std::string sub = req.matches[1];
    if (!is_safe_subpath(sub)) {
        send_error(res, 400, "unsafe path");
        return;
    }
    auto p = cfg.ui_dir / sub;
    if (!fs::exists(p) || fs::is_directory(p)) {
        send_error(res, 404, "not found");
        return;
    }
    std::string body;
    if (!read_file(p, body)) {
        send_error(res, 500, "failed to read file");
        return;
    }
    res.status = 200;
    res.set_content(body, mime_for(p));
}

void handle_get_training_data(const ServerConfig& cfg,
                              const httplib::Request&, httplib::Response& res) {
    if (!fs::exists(cfg.training_data_path)) {
        send_error(res, 404,
            "Training log not found at " + cfg.training_data_path.string()
            + ". Run benchmark or POST /api/run_training first.");
        return;
    }
    try {
        auto j = read_json_with_retry(cfg.training_data_path);
        res.status = 200;
        res.set_content(j.dump(), "application/json");
    } catch (const std::exception&) {
        send_error(res, 503, "Training log is currently being written. Retry shortly.");
    }
}

void handle_get_training_status(const ServerConfig& cfg,
                                const httplib::Request&, httplib::Response& res) {
    if (!fs::exists(cfg.training_status_path)) {
        send_error(res, 404,
            "Training status not found at " + cfg.training_status_path.string());
        return;
    }
    try {
        auto j = read_json_with_retry(cfg.training_status_path);
        res.status = 200;
        res.set_content(j.dump(), "application/json");
    } catch (const std::exception&) {
        send_error(res, 503, "Training status is currently being written. Retry shortly.");
    }
}

void handle_post_run_training(const ServerConfig& cfg,
                              const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_error(res, 422, std::string("invalid JSON body: ") + e.what());
        return;
    }
    TrainingParams p{};
    std::string err;
    if (!parse_training_params(body, p, err)) {
        send_error(res, 422, err);
        return;
    }
    auto argv = build_cmd_argv(cfg.benchmark_binary, p);

    // Best-effort placeholder.  app.py wraps in try/except OSError.
    try {
        ssns::io::write_starting_status(cfg.training_status_path, p.epochs);
    } catch (...) { /* swallow */ }

    pid_t pid = spawn_detached(argv);
    json resp = {
        {"status",  "started"},
        {"message", "Training subprocess spawned. "
                    "Refresh /api/training_data once the script completes."},
        {"pid",     static_cast<long long>(pid)},
        {"params",  body},
        {"cmd",     argv},
    };
    res.status = 200;
    res.set_content(resp.dump(), "application/json");
}

void handle_post_manual_test(const ServerConfig& cfg,
                             const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_error(res, 422, std::string("invalid JSON body: ") + e.what());
        return;
    }
    if (!body.contains("X") || !body["X"].is_array()) {
        send_error(res, 422, "missing 'X' (array of floats)");
        return;
    }
    std::optional<LoadedPair> maybe_pair;
    try {
        maybe_pair = load_trained_pair(cfg.training_data_path);
    } catch (const HTTPException& e) {
        send_error(res, e.status, e.what());
        return;
    }
    LoadedPair& pair = *maybe_pair;
    const auto& meta = pair.metadata;
    const std::size_t S_input    = meta.at("S_input").get<std::size_t>();
    const std::size_t cluster    = meta.at("cluster_size").get<std::size_t>();
    const double      dz         = meta.at("dz").get<double>();

    const auto& Xj = body["X"];
    if (Xj.size() != S_input) {
        std::ostringstream ss;
        ss << "X length " << Xj.size() << " does not match S_input=" << S_input << ".";
        send_error(res, 422, ss.str());
        return;
    }

    ssns::linalg::Matrix X(1, S_input);
    for (std::size_t i = 0; i < S_input; ++i) X(0, i) = Xj[i].get<double>();

    auto Y_T_mat = teacher_forward_sigmoid(pair.teacher, X);
    auto Y_S_mat = student_forward_sigmoid(X, pair.W1_S, pair.W2_S);

    std::vector<double> Y_T(Y_T_mat.cols()), Y_S(Y_S_mat.cols());
    for (std::size_t c = 0; c < Y_T_mat.cols(); ++c) Y_T[c] = Y_T_mat(0, c);
    for (std::size_t c = 0; c < Y_S_mat.cols(); ++c) Y_S[c] = Y_S_mat(0, c);

    auto T_ext = ssns::protocol::extract_with_indices(Y_T, static_cast<int>(cluster), dz);
    auto S_ext = ssns::protocol::extract_with_indices(Y_S, static_cast<int>(cluster), dz);

    // shared = sorted intersection of idx_T, idx_S; mismatches counts bit
    // disagreements at shared indices.  Both index lists are produced in
    // ascending order by extract_with_indices.
    std::vector<int> shared;
    int mismatches = 0;
    {
        std::size_t i = 0, j = 0;
        while (i < T_ext.indices.size() && j < S_ext.indices.size()) {
            int it = T_ext.indices[i], js = S_ext.indices[j];
            if (it == js) {
                shared.push_back(it);
                if (T_ext.bits[i] != S_ext.bits[j]) ++mismatches;
                ++i; ++j;
            } else if (it < js) ++i;
            else ++j;
        }
    }

    // Round the floats to 6 decimals to mirror app.py.
    auto round6 = [](double x) {
        return std::round(x * 1e6) / 1e6;
    };
    json Y_T_j = json::array();
    json Y_S_j = json::array();
    for (double v : Y_T) Y_T_j.push_back(round6(v));
    for (double v : Y_S) Y_S_j.push_back(round6(v));

    json out = {
        {"Y_T",            Y_T_j},
        {"Y_S",            Y_S_j},
        {"bits_T",         T_ext.bits},
        {"idx_T",          T_ext.indices},
        {"bits_S",         S_ext.bits},
        {"idx_S",          S_ext.indices},
        {"shared_indices", shared},
        {"mismatches",     mismatches},
        {"n_confident_T",  static_cast<int>(T_ext.indices.size())},
        {"n_confident_S",  static_cast<int>(S_ext.indices.size())},
        {"match",          mismatches == 0},
        {"epoch_used",     pair.last_snapshot.at("epoch").get<long>()},
    };
    res.status = 200;
    res.set_content(out.dump(), "application/json");
}

void handle_post_stress_test(const ServerConfig& cfg,
                             const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_error(res, 422, std::string("invalid JSON body: ") + e.what());
        return;
    }
    long n_trials = 200;
    std::optional<long long> seed_opt;
    if (body.contains("n_trials")) {
        if (!body["n_trials"].is_number_integer()
            && !body["n_trials"].is_number_unsigned()) {
            send_error(res, 422, "n_trials must be int");
            return;
        }
        n_trials = body["n_trials"].get<long>();
    }
    if (n_trials < 1 || n_trials > 10000) {
        send_error(res, 422, "n_trials out of range [1, 10000]");
        return;
    }
    if (body.contains("seed") && !body["seed"].is_null()) {
        if (!body["seed"].is_number_integer()
            && !body["seed"].is_number_unsigned()) {
            send_error(res, 422, "seed must be int or null");
            return;
        }
        seed_opt = body["seed"].get<long long>();
    }

    std::optional<LoadedPair> maybe_pair;
    try {
        maybe_pair = load_trained_pair(cfg.training_data_path);
    } catch (const HTTPException& e) {
        send_error(res, e.status, e.what());
        return;
    }
    LoadedPair& pair = *maybe_pair;

    const auto& meta = pair.metadata;
    const std::size_t S_input    = meta.at("S_input").get<std::size_t>();
    const std::size_t cluster    = meta.at("cluster_size").get<std::size_t>();
    const std::size_t output_dim = meta.at("output_dim").get<std::size_t>();
    const double      dz         = meta.at("dz").get<double>();
    if (cluster == 0 || output_dim % cluster != 0) {
        send_error(res, 409, "training metadata has invalid cluster_size / output_dim");
        return;
    }
    const std::size_t n_clusters = output_dim / cluster;
    const double lo_thr = 0.5 - dz;
    const double hi_thr = 0.5 + dz;

    std::mt19937_64 rng(seed_opt.value_or(static_cast<long long>(std::random_device{}())));
    std::normal_distribution<double> nd(0.0, 1.0);

    ssns::linalg::Matrix X(static_cast<std::size_t>(n_trials), S_input);
    for (std::size_t r = 0; r < X.rows(); ++r) {
        for (std::size_t c = 0; c < X.cols(); ++c) X(r, c) = nd(rng);
    }
    auto Y_T = teacher_forward_sigmoid(pair.teacher, X);
    auto Y_S = student_forward_sigmoid(X, pair.W1_S, pair.W2_S);

    std::vector<double> n_conf_T(n_trials), n_conf_S(n_trials);
    std::vector<long>   n_shared(n_trials), mismatches_v(n_trials);
    long perfect = 0, total_mismatches = 0, total_shared_bits = 0;

    for (long r = 0; r < n_trials; ++r) {
        long ct = 0, cs = 0, sh = 0, mm = 0;
        for (std::size_t k = 0; k < n_clusters; ++k) {
            double sum_T = 0.0, sum_S = 0.0;
            for (std::size_t j = 0; j < cluster; ++j) {
                sum_T += Y_T(static_cast<std::size_t>(r), k * cluster + j);
                sum_S += Y_S(static_cast<std::size_t>(r), k * cluster + j);
            }
            const double mt = sum_T / static_cast<double>(cluster);
            const double ms = sum_S / static_cast<double>(cluster);
            const bool conf_T = (mt >= hi_thr) || (mt <= lo_thr);
            const bool conf_S = (ms >= hi_thr) || (ms <= lo_thr);
            const int  bit_T  = (mt >= hi_thr) ? 1 : 0;
            const int  bit_S  = (ms >= hi_thr) ? 1 : 0;
            if (conf_T) ++ct;
            if (conf_S) ++cs;
            if (conf_T && conf_S) {
                ++sh;
                if (bit_T != bit_S) ++mm;
            }
        }
        n_conf_T[r]      = static_cast<double>(ct);
        n_conf_S[r]      = static_cast<double>(cs);
        n_shared[r]      = sh;
        mismatches_v[r]  = mm;
        if (mm == 0) ++perfect;
        total_mismatches  += mm;
        total_shared_bits += sh;
    }

    std::vector<double> shared_dbl(n_shared.begin(), n_shared.end());
    std::vector<double> mm_dbl(mismatches_v.begin(), mismatches_v.end());

    json hist = hist_counts(n_shared, /*n_bins=*/20, /*lo=*/0,
                            /*hi=*/static_cast<long>(n_clusters));

    json resp = {
        {"n_trials",          n_trials},
        {"epoch_used",        pair.last_snapshot.at("epoch").get<long>()},
        {"total_clusters",    static_cast<long>(n_clusters)},
        {"cluster_size",      static_cast<long>(cluster)},
        {"dead_zone",         dz},
        {"teacher_confident", summary(n_conf_T)},
        {"student_confident", summary(n_conf_S)},
        {"shared",            summary(shared_dbl)},
        {"mismatches",        summary(mm_dbl)},
        {"match_rate_pct",    100.0 * static_cast<double>(perfect)
                                  / static_cast<double>(n_trials)},
        {"perfect_trials",    perfect},
        {"total_mismatches",  total_mismatches},
        {"total_shared_bits", total_shared_bits},
        {"histogram_shared",  hist},
        {"seed",              seed_opt.has_value()
                              ? json(*seed_opt)
                              : json(nullptr)},
    };
    res.status = 200;
    res.set_content(resp.dump(), "application/json");
}

}  // anon namespace

// ---------------------------------------------------------------------------
// Server class.
// ---------------------------------------------------------------------------

Server::Server(ServerConfig cfg)
    : cfg_(std::move(cfg)),
      srv_(std::make_unique<httplib::Server>())
{
    auto& s = *srv_;
    s.Get("/", [this](const httplib::Request& q, httplib::Response& r){
        handle_root(cfg_, q, r);
    });
    // Static files under /ui/*.  Capture group is the remainder.
    s.Get(R"(/ui/(.+))", [this](const httplib::Request& q, httplib::Response& r){
        handle_static(cfg_, q, r);
    });
    s.Get("/api/training_data", [this](const httplib::Request& q, httplib::Response& r){
        handle_get_training_data(cfg_, q, r);
    });
    s.Get("/api/training_status", [this](const httplib::Request& q, httplib::Response& r){
        handle_get_training_status(cfg_, q, r);
    });
    s.Post("/api/run_training", [this](const httplib::Request& q, httplib::Response& r){
        handle_post_run_training(cfg_, q, r);
    });
    s.Post("/api/manual_test", [this](const httplib::Request& q, httplib::Response& r){
        handle_post_manual_test(cfg_, q, r);
    });
    s.Post("/api/stress_test", [this](const httplib::Request& q, httplib::Response& r){
        handle_post_stress_test(cfg_, q, r);
    });
}

Server::~Server() {
    stop();
}

void Server::listen(const std::string& host, int port) {
    running_ = true;
    srv_->listen(host.c_str(), port);
    running_ = false;
}

void Server::start_in_thread(const std::string& host, int port) {
    thread_ = std::thread([this, host, port]{
        running_ = true;
        srv_->listen(host.c_str(), port);
        running_ = false;
    });
}

void Server::stop() {
    if (srv_) srv_->stop();
    if (thread_.joinable()) thread_.join();
    running_ = false;
}

}  // namespace ssns::http
