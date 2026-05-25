// http server impl
// each endpoint is a free fn bound to httplib::Server held by wrapper class
//
// /api/run_training spawns subprocess with posix fork+execvp
// Это нужно потому что system() не отдаёт pid, а нам нужно знать pid для последующего /api/stop_training
// child detaches via setsid and redirects stdio to /dev/null before exec
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

        // reply with `{"detail": ...}` body and given status
        void send_error(httplib::Response &res, int status, const std::string &detail) {
            json body = {{"detail", detail}};
            res.status = status;
            res.set_content(body.dump(), "application/json");
        }

        // read json from path with up to 3 retries 50 ms backoff, tolerates torn read while another proc rewrites the file
        json read_json_with_retry(const fs::path &path,
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
                } catch (const json::parse_error &e) {
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

        // pick mime from file ext, falls back to octet-stream
        std::string mime_for(const fs::path &p) {
            auto ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
            if (ext == ".css") return "text/css; charset=utf-8";
            if (ext == ".js") return "application/javascript; charset=utf-8";
            if (ext == ".json") return "application/json; charset=utf-8";
            if (ext == ".png") return "image/png";
            if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
            if (ext == ".gif") return "image/gif";
            if (ext == ".svg") return "image/svg+xml";
            if (ext == ".ico") return "image/x-icon";
            if (ext == ".txt") return "text/plain; charset=utf-8";
            return "application/octet-stream";
        }

        // slurp whole file into out, false on open fail
        bool read_file(const fs::path &p, std::string &out) {
            std::ifstream in(p, std::ios::binary);
            if (!in) return false;
            std::ostringstream ss;
            ss << in.rdbuf();
            out = ss.str();
            return true;
        }

        // reject ".." traversal, only allow simple relative descent under ui_dir
        bool is_safe_subpath(const std::string &sub) {
            if (sub.empty()) return false;
            if (sub.find("..") != std::string::npos) return false;
            if (sub.front() == '/') return false;
            return true;
        }

        // ----- /api/run_training subprocess spawn ----------------------------------

        // fields expected in POST body of /api/run_training
        struct TrainingParams {
            long T_input, T_hidden, S_input, S_hidden;
            long output_dim, cluster_size, batch_size, epochs;
            double dz, lr_max, warmup_frac;
            long samples_to_log = 20;
            long snapshot_count = 10;
            bool use_fhe = false;
            double bimodality_alpha = 0.0; // 0 disables (default)
            double simulate_fhe_noise = 0.0; // use_fhe=false + this >0 = plain + gaussian noise (fast proxy)
            bool key_confirmation = false; // when true append --key-confirmation N after training
            long key_confirmation_trials = 10000;
            bool seed_override = false; // set if UI provided explicit seeds, otherwise keep CLI defaults
            std::uint64_t teacher_seed = 0;
            std::uint64_t bfa_seed = 0;
        };

        // parse + validate POST body, on fail err describes missing or bad field returns false
        bool parse_training_params(const json &body, TrainingParams &out, std::string &err) {
            auto need_int = [&](const char *name, long &dst) -> bool {
                if (!body.contains(name)) {
                    err = std::string("missing field: ") + name;
                    return false;
                }
                const auto &v = body.at(name);
                if (!v.is_number_integer() && !v.is_number_unsigned()) {
                    err = std::string("field must be int: ") + name;
                    return false;
                }
                dst = v.get<long>();
                return true;
            };
            auto need_double = [&](const char *name, double &dst) -> bool {
                if (!body.contains(name)) {
                    err = std::string("missing field: ") + name;
                    return false;
                }
                const auto &v = body.at(name);
                if (!v.is_number()) {
                    err = std::string("field must be number: ") + name;
                    return false;
                }
                dst = v.get<double>();
                return true;
            };

            if (!body.is_object()) {
                err = "request body must be a JSON object";
                return false;
            }
            if (!need_int("T_input", out.T_input)) return false;
            if (!need_int("T_hidden", out.T_hidden)) return false;
            if (!need_int("S_input", out.S_input)) return false;
            if (!need_int("S_hidden", out.S_hidden)) return false;
            if (!need_int("output_dim", out.output_dim)) return false;
            if (!need_int("cluster_size", out.cluster_size)) return false;
            if (!need_int("batch_size", out.batch_size)) return false;
            if (!need_int("epochs", out.epochs)) return false;
            if (!need_double("dz", out.dz)) return false;
            if (!need_double("lr_max", out.lr_max)) return false;
            if (!need_double("warmup_frac", out.warmup_frac)) return false;

            if (body.contains("samples_to_log")) {
                if (!body["samples_to_log"].is_number_integer()
                    && !body["samples_to_log"].is_number_unsigned()) {
                    err = "samples_to_log must be int";
                    return false;
                }
                out.samples_to_log = body["samples_to_log"].get<long>();
                if (out.samples_to_log < 1 || out.samples_to_log > 1024) {
                    err = "samples_to_log out of range [1,1024]";
                    return false;
                }
            }
            if (body.contains("snapshot_count")) {
                if (!body["snapshot_count"].is_number_integer()
                    && !body["snapshot_count"].is_number_unsigned()) {
                    err = "snapshot_count must be int";
                    return false;
                }
                out.snapshot_count = body["snapshot_count"].get<long>();
                if (out.snapshot_count < 2 || out.snapshot_count > 200) {
                    err = "snapshot_count out of range [2,200]";
                    return false;
                }
            }
            if (body.contains("bimodality_alpha")) {
                if (!body["bimodality_alpha"].is_number()) {
                    err = "bimodality_alpha must be number";
                    return false;
                }
                out.bimodality_alpha = body["bimodality_alpha"].get<double>();
                if (out.bimodality_alpha < 0.0 || out.bimodality_alpha > 5.0) {
                    err = "bimodality_alpha out of range [0,5]";
                    return false;
                }
            }
            if (body.contains("simulate_fhe_noise")) {
                if (!body["simulate_fhe_noise"].is_number()) {
                    err = "simulate_fhe_noise must be number";
                    return false;
                }
                out.simulate_fhe_noise = body["simulate_fhe_noise"].get<double>();
                if (out.simulate_fhe_noise < 0.0 || out.simulate_fhe_noise > 10.0) {
                    err = "simulate_fhe_noise out of range [0, 10]";
                    return false;
                }
            }
            if (body.contains("key_confirmation")) {
                if (!body["key_confirmation"].is_boolean()) {
                    err = "key_confirmation must be bool";
                    return false;
                }
                out.key_confirmation = body["key_confirmation"].get<bool>();
            }
            if (body.contains("key_confirmation_trials")) {
                if (!body["key_confirmation_trials"].is_number_integer()
                    && !body["key_confirmation_trials"].is_number_unsigned()) {
                    err = "key_confirmation_trials must be int";
                    return false;
                }
                out.key_confirmation_trials = body["key_confirmation_trials"].get<long>();
                if (out.key_confirmation_trials < 1 || out.key_confirmation_trials > 1000000) {
                    err = "key_confirmation_trials out of range [1, 1000000]";
                    return false;
                }
            }
            if (body.contains("use_fhe")) {
                if (!body["use_fhe"].is_boolean()) {
                    err = "use_fhe must be bool";
                    return false;
                }
                out.use_fhe = body["use_fhe"].get<bool>();
            }
            // optional explicit seeds, must come as a pair if provided
            auto need_seed = [&](const char *name, std::uint64_t &dst) -> bool {
                if (!body.contains(name)) return true;
                const auto &v = body.at(name);
                if (!v.is_number_integer() && !v.is_number_unsigned()) {
                    err = std::string("field must be uint: ") + name;
                    return false;
                }
                long long signed_val = v.get<long long>();
                if (signed_val < 0) {
                    err = std::string("field must be non-negative: ") + name;
                    return false;
                }
                dst = static_cast<std::uint64_t>(signed_val);
                out.seed_override = true;
                return true;
            };
            if (!need_seed("teacher_seed", out.teacher_seed)) return false;
            if (!need_seed("bfa_seed", out.bfa_seed)) return false;
            return true;
        }

        // build argv vec for benchmark subproc
        // first entry is benchmark binary path becomes argv[0]
        // --output-path и --status-path форсим явно, иначе subprocess пишет в ui_data/... относительно своего CWD, и /api эти файлы не находит
        std::vector<std::string> build_cmd_argv(const fs::path &benchmark,
                                                const TrainingParams &p,
                                                const fs::path &output_path,
                                                const fs::path &status_path) {
            std::vector<std::string> a;
            a.push_back(benchmark.string());
            auto push = [&](const char *flag, const std::string &v) {
                a.emplace_back(flag);
                a.emplace_back(v);
            };
            push("--t-input", std::to_string(p.T_input));
            push("--t-hidden", std::to_string(p.T_hidden));
            push("--s-input", std::to_string(p.S_input));
            push("--s-hidden", std::to_string(p.S_hidden));
            push("--output-dim", std::to_string(p.output_dim));
            push("--cluster-size", std::to_string(p.cluster_size));
            push("--batch-size", std::to_string(p.batch_size));
            push("--epochs", std::to_string(p.epochs)); {
                std::ostringstream ss;
                ss << p.dz;
                push("--dz", ss.str());
            } {
                std::ostringstream ss;
                ss << p.lr_max;
                push("--lr-max", ss.str());
            } {
                std::ostringstream ss;
                ss << p.warmup_frac;
                push("--warmup-frac", ss.str());
            }
            push("--samples-to-log", std::to_string(p.samples_to_log));

            long snap_count = std::max<long>(1L, p.snapshot_count);
            long snap_interval = std::max<long>(1L, p.epochs / snap_count);
            push("--snapshot-interval", std::to_string(snap_interval));

            push("--output-path", output_path.string());
            push("--status-path", status_path.string());

            if (p.bimodality_alpha > 0.0) {
                std::ostringstream ss;
                ss << p.bimodality_alpha;
                push("--bimodality-alpha", ss.str());
            }
            if (!p.use_fhe && p.simulate_fhe_noise > 0.0) {
                std::ostringstream ss;
                ss << p.simulate_fhe_noise;
                push("--simulate-fhe-noise", ss.str());
            }
            if (p.key_confirmation) {
                push("--key-confirmation", std::to_string(p.key_confirmation_trials));
            }

            if (p.use_fhe) a.emplace_back("--use-fhe");

            if (p.seed_override) {
                push("--teacher-seed", std::to_string(p.teacher_seed));
                push("--bfa-seed", std::to_string(p.bfa_seed));
            }
            return a;
        }

#if defined(__unix__) || defined(__APPLE__)

        // spawn argv[0] detached via fork+execvp
        // child runs setsid and redirects stdio to /dev/null before exec
        // returns child pid or -1 if fork failed
        pid_t spawn_detached(const std::vector<std::string> &argv) {
            if (argv.empty()) return -1;

            // avoid zombies, install SIG_IGN for SIGCHLD on first call so kids get auto reaped
            // idempotent and harmless
            static bool sigchld_installed = false;
            if (!sigchld_installed) {
                struct sigaction sa{};
                sa.sa_handler = SIG_IGN;
                sa.sa_flags = SA_NOCLDWAIT;
                sigemptyset(&sa.sa_mask);
                sigaction(SIGCHLD, &sa, nullptr);
                sigchld_installed = true;
            }

            pid_t pid = fork();
            if (pid < 0) return -1;
            if (pid == 0) {
                // child
                setsid();
                int dn = ::open("/dev/null", O_RDWR);
                if (dn >= 0) {
                    dup2(dn, STDIN_FILENO);
                    dup2(dn, STDOUT_FILENO);
                    dup2(dn, STDERR_FILENO);
                    if (dn > STDERR_FILENO) ::close(dn);
                }
                // build argv as char* array for execvp
                std::vector<char *> raw;
                raw.reserve(argv.size() + 1);
                for (auto &s: argv) {
                    raw.push_back(const_cast<char *>(s.c_str()));
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

        // nested 2d json array -> ssns::linalg::Matrix
        ssns::linalg::Matrix matrix_from_json(const json &m) {
            if (!m.is_array() || m.empty() || !m.front().is_array()) {
                throw std::runtime_error("expected 2-D JSON array");
            }
            const std::size_t rows = m.size();
            const std::size_t cols = m.front().size();
            ssns::linalg::Matrix M(rows, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                const auto &row = m[r];
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

        // carries http status through normal exception flow so handler stays small
        struct HTTPException : public std::runtime_error {
            int status;

            HTTPException(int s, const std::string &m) : std::runtime_error(m), status(s) {
            }
        };

        // load latest snapshot of {Teacher Student} from data_path
        // throws HTTPException with 404/409/503 on common fail modes
        // Валидируем до построения Teacher: иначе пустой/корявый файл даёт 500 уже внутри ssns::nn::Teacher, именно поэтому здесь так много 409
        LoadedPair load_trained_pair(const fs::path &data_path) {
            if (!fs::exists(data_path)) {
                throw HTTPException(404,
                                    "Training log not found at " + data_path.string()
                                    + ". Run training first.");
            }
            json data;
            try {
                data = read_json_with_retry(data_path);
            } catch (const std::exception &) {
                throw HTTPException(503,
                                    "Training log is currently being written. Retry shortly.");
            }
            if (!data.contains("metadata") || !data.contains("snapshots")) {
                throw HTTPException(409, "Training log missing 'metadata' or 'snapshots'.");
            }
            const auto &meta = data["metadata"];
            const auto &snaps = data["snapshots"];
            if (!snaps.is_array() || snaps.empty()) {
                throw HTTPException(409,
                                    "Training log has no snapshots yet - wait for training to "
                                    "produce at least one snapshot.");
            }
            if (!meta.contains("S_input") || meta["S_input"].is_null()) {
                throw HTTPException(409, "Training log metadata missing 'S_input'.");
            }
            if (!meta.contains("teacher_seed") || meta["teacher_seed"].is_null()) {
                throw HTTPException(409,
                                    "Training log lacks 'teacher_seed' metadata - re-run training "
                                    "with the latest benchmark.");
            }
            const std::size_t T_input = meta.at("T_input").get<std::size_t>();
            const std::size_t T_hidden = meta.at("T_hidden").get<std::size_t>();
            const std::size_t output_dim = meta.at("output_dim").get<std::size_t>();
            const std::uint64_t teacher_seed = meta.at("teacher_seed").get<std::uint64_t>();

            ssns::nn::Teacher teacher(T_input, T_hidden, output_dim, teacher_seed);
            const auto &last = snaps.back();
            auto W1_S = matrix_from_json(last.at("weights").at("W1"));
            auto W2_S = matrix_from_json(last.at("weights").at("W2"));
            return LoadedPair{meta, last, std::move(teacher), std::move(W1_S), std::move(W2_S)};
        }

        // 1d summary stats mean std (population) min max median
        json summary(const std::vector<double> &v) {
            if (v.empty()) {
                return json{{"mean", 0.0}, {"std", 0.0}, {"min", 0.0}, {"max", 0.0}, {"median", 0.0}};
            }
            double mn = v[0], mx = v[0], sum = 0.0;
            for (double x: v) {
                mn = std::min(mn, x);
                mx = std::max(mx, x);
                sum += x;
            }
            const double mean = sum / static_cast<double>(v.size());
            double var = 0.0;
            for (double x: v) var += (x - mean) * (x - mean);
            var /= static_cast<double>(v.size()); // population variance unbiased=False
            auto sorted = v;
            std::sort(sorted.begin(), sorted.end());
            double median;
            const std::size_t n = sorted.size();
            if (n % 2 == 1) median = sorted[n / 2];
                // for even n pick lower middle to match torch.tensor.median
            else median = sorted[n / 2 - 1];
            return json{
                {"mean", mean},
                {"std", std::sqrt(var)},
                {"min", mn},
                {"max", mx},
                {"median", median},
            };
        }

        // fixed bin histogram returns `{"bins": [edges...], "counts": [...]}`
        json hist_counts(const std::vector<long> &values, int n_bins,
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
            for (long v: values) {
                long idx = (width > 0)
                               ? static_cast<long>((v - lo) / width)
                               : 0L;
                if (idx >= n_bins) idx = n_bins - 1;
                if (idx < 0) idx = 0;
                counts[idx]++;
            }
            return json{{"bins", edges}, {"counts", counts}};
        }

        // run student forward Y = sigmoid(ReLU(X @ W1) @ W2) on batched X shape [n, S_input], uses c++ matrix kernels
        ssns::linalg::Matrix student_forward_sigmoid(
            const ssns::linalg::Matrix &X,
            const ssns::linalg::Matrix &W1,
            const ssns::linalg::Matrix &W2) {
            auto pre = ssns::linalg::matmul(X, W1);
            auto post = ssns::nn::relu(pre);
            auto raw = ssns::linalg::matmul(post, W2);
            return ssns::nn::sigmoid(raw);
        }

        // teacher forward Y = sigmoid(Teacher(X))
        ssns::linalg::Matrix teacher_forward_sigmoid(
            const ssns::nn::Teacher &teacher,
            const ssns::linalg::Matrix &X) {
            auto raw = teacher.forward(X);
            return ssns::nn::sigmoid(raw);
        }

        // ----- endpoint handlers ---------------------------------------------------

        // GET / serves ui/index.html or 404 if missing
        void handle_root(const ServerConfig &cfg, const httplib::Request &, httplib::Response &res) {
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

        // GET /ui/<path> serves a static file rooted at ui_dir
        // rejects unsafe paths (".." traversal leading /)
        void handle_static(const ServerConfig &cfg, const httplib::Request &req,
                           httplib::Response &res) {
            // matches.size() == 2 means full match + one capture group
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

        // GET /api/training_data returns training log json
        // 404 if missing, 503 if currently rewritten, else 200
        void handle_get_training_data(const ServerConfig &cfg,
                                      const httplib::Request &, httplib::Response &res) {
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
            } catch (const std::exception &) {
                send_error(res, 503, "Training log is currently being written. Retry shortly.");
            }
        }

        // GET /api/training_status returns live status json
        // 404 if missing, 503 if currently rewritten, else 200
        void handle_get_training_status(const ServerConfig &cfg,
                                        const httplib::Request &, httplib::Response &res) {
            if (!fs::exists(cfg.training_status_path)) {
                send_error(res, 404,
                           "Training status not found at " + cfg.training_status_path.string());
                return;
            }
            try {
                auto j = read_json_with_retry(cfg.training_status_path);
                res.status = 200;
                res.set_content(j.dump(), "application/json");
            } catch (const std::exception &) {
                send_error(res, 503, "Training status is currently being written. Retry shortly.");
            }
        }

        // POST /api/run_training validates body spawns ssns-benchmark detached returns 200 + `{pid params cmd}`
        // 422 on bad body 500 if benchmark binary missing
        void handle_post_run_training(const ServerConfig &cfg, const httplib::Request &req, httplib::Response &res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception &e) {
                send_error(res, 422, std::string("invalid JSON body: ") + e.what());
                return;
            }
            TrainingParams p{};
            std::string err;
            if (!parse_training_params(body, p, err)) {
                send_error(res, 422, err);
                return;
            }
            // pre fork existence check
            // if benchmark binary missing or not exec we return 500 right away
            // else child dies on execvp ENOENT, parent doesnt notice, user stares at "epoch 0 / N" forever
            {
                std::error_code ec;
                const bool ok =
                        std::filesystem::exists(cfg.benchmark_binary, ec) && !ec
                        && (::access(cfg.benchmark_binary.c_str(), X_OK) == 0);
                if (!ok) {
                    send_error(res, 500,
                               "benchmark binary not found or not executable at " +
                               cfg.benchmark_binary.string() +
                               ". Build ssns-benchmark and/or restart ssns-server with "
                               "--benchmark <path>.");
                    return;
                }
            }

            auto argv = build_cmd_argv(cfg.benchmark_binary, p,
                cfg.training_data_path, cfg.training_status_path);

            // best effort write of running:true placeholder
            // failures swallowed, subproc overwrites file shortly
            try {
                ssns::io::write_starting_status(cfg.training_status_path, p.epochs);
            } catch (...) {
                /* swallow */
            }

            pid_t pid = spawn_detached(argv);
            json resp = {
                {"status", "started"},
                {
                    "message", "Training subprocess spawned. "
                    "Refresh /api/training_data once the script completes."
                },
                {"pid", static_cast<long long>(pid)},
                {"params", body},
                {"cmd", argv},
            };
            res.status = 200;
            res.set_content(resp.dump(), "application/json");
        }

        // POST /api/stop_training body `{"pid": int}`
        // confirms pid is a benchmark via /proc then sends sigterm
        // always rewrites training_status json with running=false stopped=true so frontend poller settles even if child got sigkilled mid write
        // 422 on bad body 404 if pid not a benchmark 200 on success
        void handle_post_stop_training(const ServerConfig &cfg,
                                       const httplib::Request &req, httplib::Response &res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception &e) {
                send_error(res, 422, std::string("invalid JSON body: ") + e.what());
                return;
            }
            if (!body.contains("pid") || !body["pid"].is_number_integer()) {
                send_error(res, 422, "missing or invalid 'pid' field (integer expected)");
                return;
            }
            pid_t pid = static_cast<pid_t>(body["pid"].get<long long>());
            if (pid <= 1) {
                send_error(res, 422, "pid must be > 1");
                return;
            }

            // sanity check confirm pid is a benchmark proc
            // accept /proc/<pid>/comm == "ssns-benchmark" or any cmdline that contains the bin basename
            bool looks_like_benchmark = false; {
                std::ifstream comm_in("/proc/" + std::to_string(pid) + "/comm");
                std::string comm;
                if (comm_in && std::getline(comm_in, comm)) {
                    // linux truncates comm to 15 chars, "ssns-benchmark" is 14, fits
                    if (comm.find("ssns-benchmark") != std::string::npos) {
                        looks_like_benchmark = true;
                    }
                }
                if (!looks_like_benchmark) {
                    std::ifstream cmd_in("/proc/" + std::to_string(pid) + "/cmdline");
                    std::string cmdline;
                    if (cmd_in) {
                        std::ostringstream oss;
                        oss << cmd_in.rdbuf();
                        cmdline = oss.str();
                        if (cmdline.find("ssns-benchmark") != std::string::npos) {
                            looks_like_benchmark = true;
                        }
                    }
                }
            }
            if (!looks_like_benchmark) {
                send_error(res, 404, "pid does not refer to a running ssns-benchmark process");
                return;
            }

            int kill_rc = ::kill(pid, SIGTERM);
            int saved_errno = errno;

            // read current status if any so we keep epoch/loss/elapsed values rather than zero them
            // reads guarded against missing key + type mismatch so a corrupt status file cant take endpoint down
            json existing;
            try {
                std::ifstream in(cfg.training_status_path);
                if (in) in >> existing;
            } catch (...) { existing = json::object(); }
            if (!existing.is_object()) existing = json::object();

            auto get_or = [&](const char *key, auto fallback) -> json {
                auto it = existing.find(key);
                if (it == existing.end()) return json(fallback);
                return *it;
            };

            // format current utc time as iso-8601 with trailing Z
            const auto now_t = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now_t);
            std::tm tm_buf{};
            gmtime_r(&tt, &tm_buf);
            char ts_buf[64];
            std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);

            json final_status = {
                {"running", false},
                {"epoch", get_or("epoch", 0)},
                {"total_epochs", get_or("total_epochs", 0)},
                {"loss", get_or("loss", 0.0)},
                {"lr", get_or("lr", 0.0)},
                {"started_at", get_or("started_at", json())},
                {"elapsed_sec", get_or("elapsed_sec", 0.0)},
                {"eta_sec", json()},
                {"completed_at", std::string(ts_buf)},
                {"stopped", true},
            };
            try {
                if (cfg.training_status_path.has_parent_path()) {
                    std::filesystem::create_directories(cfg.training_status_path.parent_path());
                }
                std::ofstream out(cfg.training_status_path);
                out << final_status.dump();
            } catch (...) {
                /* swallow kill already sent response is still valid */
            }

            json resp = {
                {"status", kill_rc == 0 ? "stopped" : "kill_failed"},
                {"pid", static_cast<long long>(pid)},
                {"signal", "SIGTERM"},
                {"errno", kill_rc == 0 ? 0 : saved_errno},
            };
            res.status = 200;
            res.set_content(resp.dump(), "application/json");
        }

        // POST /api/manual_test body `{"X": [...]}` length S_input
        // loads latest T+S snapshot runs both forwards on X extracts confident bits returns side by side bits + shared idx
        // 422 on bad body 404/409/503 if no usable training log
        void handle_post_manual_test(const ServerConfig &cfg,
                                     const httplib::Request &req, httplib::Response &res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception &e) {
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
            } catch (const HTTPException &e) {
                send_error(res, e.status, e.what());
                return;
            }
            LoadedPair &pair = *maybe_pair;
            const auto &meta = pair.metadata;
            const std::size_t S_input = meta.at("S_input").get<std::size_t>();
            const std::size_t cluster = meta.at("cluster_size").get<std::size_t>();
            const double dz = meta.at("dz").get<double>();

            const auto &Xj = body["X"];
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

            // shared is sorted intersection of idx_T and idx_S
            // mismatches counts bit disagreements at shared idx
            // both index lists already ascending from extract_with_indices
            std::vector<int> shared;
            int mismatches = 0; {
                std::size_t i = 0, j = 0;
                while (i < T_ext.indices.size() && j < S_ext.indices.size()) {
                    int it = T_ext.indices[i], js = S_ext.indices[j];
                    if (it == js) {
                        shared.push_back(it);
                        if (T_ext.bits[i] != S_ext.bits[j]) ++mismatches;
                        ++i;
                        ++j;
                    } else if (it < js) ++i;
                    else ++j;
                }
            }

            // round floats to 6 decimals before sending to ui
            auto round6 = [](double x) {
                return std::round(x * 1e6) / 1e6;
            };
            json Y_T_j = json::array();
            json Y_S_j = json::array();
            for (double v: Y_T) Y_T_j.push_back(round6(v));
            for (double v: Y_S) Y_S_j.push_back(round6(v));

            json out = {
                {"Y_T", Y_T_j},
                {"Y_S", Y_S_j},
                {"bits_T", T_ext.bits},
                {"idx_T", T_ext.indices},
                {"bits_S", S_ext.bits},
                {"idx_S", S_ext.indices},
                {"shared_indices", shared},
                {"mismatches", mismatches},
                {"n_confident_T", static_cast<int>(T_ext.indices.size())},
                {"n_confident_S", static_cast<int>(S_ext.indices.size())},
                {"match", mismatches == 0},
                {"epoch_used", pair.last_snapshot.at("epoch").get<long>()},
            };
            res.status = 200;
            res.set_content(out.dump(), "application/json");
        }

        // POST /api/stress_test body `{"n_trials": int?, "seed": int?}`
        // runs N rand trials of T/S bit extraction
        // returns summary stats per trial shared bits histogram perfect match count total mm
        // 422 on bad body 404/409/503 if no usable training log
        void handle_post_stress_test(const ServerConfig &cfg,
                                     const httplib::Request &req, httplib::Response &res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception &e) {
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
            if (n_trials < 1) {
                send_error(res, 422, "n_trials must be >= 1");
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
            } catch (const HTTPException &e) {
                send_error(res, e.status, e.what());
                return;
            }
            LoadedPair &pair = *maybe_pair;

            const auto &meta = pair.metadata;
            const std::size_t S_input = meta.at("S_input").get<std::size_t>();
            const std::size_t cluster = meta.at("cluster_size").get<std::size_t>();
            const std::size_t output_dim = meta.at("output_dim").get<std::size_t>();
            const double dz = meta.at("dz").get<double>();
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
            std::vector<long> n_shared(n_trials), mismatches_v(n_trials);
            long perfect = 0, total_mismatches = 0, total_shared_bits = 0;
            // sha-256 confirmation simulation per trial
            // dropped:     hex_key(bits_T) != hex_key(bits_S) protocol would retry
            // silent_fail: hashes match but bits differ sha-256 collision astronomically rare
            long hash_dropped = 0, hash_silent_fail = 0;

            std::vector<int> bits_T_shared, bits_S_shared;
            bits_T_shared.reserve(n_clusters);
            bits_S_shared.reserve(n_clusters);

            for (long r = 0; r < n_trials; ++r) {
                long ct = 0, cs = 0, sh = 0, mm = 0;
                bits_T_shared.clear();
                bits_S_shared.clear();
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
                    const int bit_T = (mt >= hi_thr) ? 1 : 0;
                    const int bit_S = (ms >= hi_thr) ? 1 : 0;
                    if (conf_T) ++ct;
                    if (conf_S) ++cs;
                    if (conf_T && conf_S) {
                        ++sh;
                        bits_T_shared.push_back(bit_T);
                        bits_S_shared.push_back(bit_S);
                        if (bit_T != bit_S) ++mm;
                    }
                }
                n_conf_T[r] = static_cast<double>(ct);
                n_conf_S[r] = static_cast<double>(cs);
                n_shared[r] = sh;
                mismatches_v[r] = mm;
                if (mm == 0) ++perfect;
                total_mismatches += mm;
                total_shared_bits += sh;

                // real sha-256 hash exchange both parties hash their shared bits and compare digests
                // экспериментально hash_dropped должен совпасть с (n_trials - perfect)
                const std::string hT = ssns::protocol::hex_key(bits_T_shared);
                const std::string hS = ssns::protocol::hex_key(bits_S_shared);
                if (hT != hS) {
                    ++hash_dropped;
                } else if (bits_T_shared != bits_S_shared) {
                    ++hash_silent_fail;
                }
            }

            std::vector<double> shared_dbl(n_shared.begin(), n_shared.end());
            std::vector<double> mm_dbl(mismatches_v.begin(), mismatches_v.end());

            json hist = hist_counts(n_shared, /*n_bins=*/20, /*lo=*/0,
                                    /*hi=*/static_cast<long>(n_clusters));

            json resp = {
                {"n_trials", n_trials},
                {"epoch_used", pair.last_snapshot.at("epoch").get<long>()},
                {"total_clusters", static_cast<long>(n_clusters)},
                {"cluster_size", static_cast<long>(cluster)},
                {"dead_zone", dz},
                {"teacher_confident", summary(n_conf_T)},
                {"student_confident", summary(n_conf_S)},
                {"shared", summary(shared_dbl)},
                {"mismatches", summary(mm_dbl)},
                {
                    "match_rate_pct", 100.0 * static_cast<double>(perfect)
                                      / static_cast<double>(n_trials)
                },
                {"perfect_trials", perfect},
                {"total_mismatches", total_mismatches},
                {"total_shared_bits", total_shared_bits},
                {"hash_dropped", hash_dropped},
                {"hash_accepted", n_trials - hash_dropped},
                {"hash_silent_fail", hash_silent_fail},
                {"histogram_shared", hist},
                {
                    "seed", seed_opt.has_value()
                                ? json(*seed_opt)
                                : json(nullptr)
                },
            };
            res.status = 200;
            res.set_content(resp.dump(), "application/json");
        }
    } // anon namespace

    // ---------------------------------------------------------------------------
    // Server class
    // ---------------------------------------------------------------------------

    // wire each route to its handler
    Server::Server(ServerConfig cfg)
        : cfg_(std::move(cfg)),
          srv_(std::make_unique<httplib::Server>()) {
        auto &s = *srv_;
        s.Get("/", [this](const httplib::Request &q, httplib::Response &r) {
            handle_root(cfg_, q, r);
        });
        // static files under /ui/* capture group is path remainder
        s.Get(R"(/ui/(.+))", [this](const httplib::Request &q, httplib::Response &r) {
            handle_static(cfg_, q, r);
        });
        s.Get("/api/training_data", [this](const httplib::Request &q, httplib::Response &r) {
            handle_get_training_data(cfg_, q, r);
        });
        s.Get("/api/training_status", [this](const httplib::Request &q, httplib::Response &r) {
            handle_get_training_status(cfg_, q, r);
        });
        s.Post("/api/run_training", [this](const httplib::Request &q, httplib::Response &r) {
            handle_post_run_training(cfg_, q, r);
        });
        s.Post("/api/stop_training", [this](const httplib::Request &q, httplib::Response &r) {
            handle_post_stop_training(cfg_, q, r);
        });
        s.Post("/api/manual_test", [this](const httplib::Request &q, httplib::Response &r) {
            handle_post_manual_test(cfg_, q, r);
        });
        s.Post("/api/stress_test", [this](const httplib::Request &q, httplib::Response &r) {
            handle_post_stress_test(cfg_, q, r);
        });
    }

    Server::~Server() {
        stop();
    }

    void Server::listen(const std::string &host, int port) {
        srv_->listen(host.c_str(), port);
    }

    void Server::start_in_thread(const std::string &host, int port) {
        thread_ = std::thread([this, host, port] {
            srv_->listen(host.c_str(), port);
        });
    }

    void Server::stop() {
        if (srv_) srv_->stop();
        if (thread_.joinable()) thread_.join();
    }
} // namespace ssns::http
