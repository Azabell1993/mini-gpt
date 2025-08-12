#include <crow.h>
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <iostream>
#include "advanced_gpt.hpp"

using namespace gpt;

// 전역 모델 인스턴스
std::unique_ptr<AdvancedGPT> global_model;

// 모델 초기화 함수
void initialize_model() {
    if (!global_model) {
        // 소형 GPT 설정 (데모용)
        int vocab_size = 1000;   // 소형으로 축소
        int d_model = 128;       // 소형으로 축소
        int n_layers = 2;        // 소형으로 축소
        int n_heads = 4;
        int d_ff = 512;          // 소형으로 축소
        int max_seq_len = 64;
        
        global_model = std::make_unique<AdvancedGPT>(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len);
        std::cout << "Advanced GPT model initialized!" << std::endl;
    }
}

// 간단한 토크나이저 (데모용)
class SimpleTokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    int next_id_;

public:
    SimpleTokenizer() : next_id_(2) {  // 0: PAD, 1: EOS
        vocab_["<PAD>"] = 0;
        vocab_["<EOS>"] = 1;
        reverse_vocab_[0] = "<PAD>";
        reverse_vocab_[1] = "<EOS>";
        
        // 일반적인 단어들 미리 추가
        std::vector<std::string> common_words = {
            "hello", "world", "the", "a", "an", "is", "are", "was", "were",
            "i", "you", "he", "she", "it", "we", "they", "and", "or", "but",
            "to", "from", "in", "on", "at", "for", "with", "by", "about"
        };
        
        for (const auto& word : common_words) {
            vocab_[word] = next_id_;
            reverse_vocab_[next_id_] = word;
            next_id_++;
        }
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // 소문자로 변환
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            
            if (vocab_.find(word) == vocab_.end()) {
                if (next_id_ < 1000) {  // vocab_size 제한
                    vocab_[word] = next_id_;
                    reverse_vocab_[next_id_] = word;
                    next_id_++;
                } else {
                    // 알 수 없는 토큰은 랜덤 ID 할당
                    tokens.push_back(rand() % 100 + 100);
                    continue;
                }
            }
            tokens.push_back(vocab_[word]);
        }
        
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int token : tokens) {
            if (reverse_vocab_.find(token) != reverse_vocab_.end()) {
                if (!result.empty()) result += " ";
                result += reverse_vocab_[token];
            }
        }
        return result;
    }
};

// 전역 토크나이저
SimpleTokenizer global_tokenizer;

// Minimal helpers: softmax over last dimension
static std::vector<double> softmax(const std::vector<double>& x) {
    std::vector<double> y(x.size());
    double m = x[0];
    for(double v: x) if(v>m) m = v;
    double sum = 0.0;
    for(size_t i=0;i<x.size();++i){ y[i] = std::exp(x[i]-m); sum += y[i]; }
    for(size_t i=0;i<x.size();++i){ y[i] /= (sum>0?sum:1.0); }
    return y;
}

// Computes y = softmax(q K^T / sqrt(dk)) V
// Shapes (demo, single-head, single-batch):
// q: [L, dk], k: [L, dk], v: [L, dv]  -> out: [L, dv]
static crow::json::wvalue attention_demo(
    const std::vector<std::vector<double>>& q,
    const std::vector<std::vector<double>>& k,
    const std::vector<std::vector<double>>& v,
    bool causal)
{
    const int L = (int)q.size();
    const int dk = (int)q[0].size();
    const int dv = (int)v[0].size();
    const double scale = 1.0/std::sqrt((double)dk);

    // scores [L,L]
    std::vector<std::vector<double>> scores(L, std::vector<double>(L, 0.0));
    for(int t=0;t<L;++t){
        for(int u=0;u<L;++u){
            double dot = 0.0;
            for(int i=0;i<dk;++i) dot += q[t][i]*k[u][i];
            scores[t][u] = dot * scale;
            if(causal && u>t) scores[t][u] = -1e9; // upper-triangular mask
        }
    }

    // softmax row-wise -> attn [L,L]
    std::vector<std::vector<double>> attn(L, std::vector<double>(L, 0.0));
    for(int t=0;t<L;++t){
        attn[t] = softmax(scores[t]);
    }

    // out = attn @ v -> [L,dv]
    std::vector<std::vector<double>> out(L, std::vector<double>(dv, 0.0));
    for(int t=0;t<L;++t){
        for(int j=0;j<dv;++j){
            double s = 0.0;
            for(int u=0;u<L;++u) s += attn[t][u] * v[u][j];
            out[t][j] = s;
        }
    }

    crow::json::wvalue W;
    W["scores"] = scores;
    W["attn"] = attn;
    W["out"] = out;
    return W;
}

int main(int argc, char** argv){
    crow::SimpleApp app;

    CROW_ROUTE(app, "/").methods(crow::HTTPMethod::GET)
    ([](){ return "mini-gpt back: see /api/health"; });

    CROW_ROUTE(app, "/favicon.ico").methods(crow::HTTPMethod::GET)
    ([](){ return crow::response(204); });

    CROW_ROUTE(app, "/api/health").methods(crow::HTTPMethod::GET)
    ([](){
        crow::json::wvalue W;
        W["ok"] = true;
        W["name"] = "mini-gpt back";
        return W;
    });

    CROW_ROUTE(app, "/api/config").methods(crow::HTTPMethod::GET)
    ([](){
        initialize_model(); // 모델 초기화 확인
        crow::json::wvalue W;
        W["vocab_size"] = 1000;
        W["n_layers"] = 2;
        W["n_heads"] = 4;
        W["d_model"] = 128;
        W["d_ff"] = 512;
        W["max_seq_len"] = 64;
        W["dropout"] = 0.1;
        W["pre_ln"] = true;
        W["features"] = std::vector<std::string>{"flash_attention", "kv_cache", "advanced_sampling"};
        return W;
    });

    // POST /api/attention  { q:[[...],...], k:[[...],...], v:[[...],...], causal:true }
    CROW_ROUTE(app, "/api/attention").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req){
        auto body = crow::json::load(req.body);
        if(!body){
            return crow::response(400, "invalid json");
        }
        auto Qj = body["q"];
        auto Kj = body["k"];
        auto Vj = body["v"];
        bool causal = body.has("causal") ? (bool)body["causal"].b() : true;

        std::vector<std::vector<double>> q, k, v;
        for(size_t i=0;i<Qj.size();++i){
            std::vector<double> row;
            for(size_t j=0;j<Qj[i].size();++j) row.push_back((double)Qj[i][j].d());
            q.push_back(std::move(row));
        }
        for(size_t i=0;i<Kj.size();++i){
            std::vector<double> row;
            for(size_t j=0;j<Kj[i].size();++j) row.push_back((double)Kj[i][j].d());
            k.push_back(std::move(row));
        }
        for(size_t i=0;i<Vj.size();++i){
            std::vector<double> row;
            for(size_t j=0;j<Vj[i].size();++j) row.push_back((double)Vj[i][j].d());
            v.push_back(std::move(row));
        }

        auto W = attention_demo(q,k,v,causal);
        crow::response r;
        r.code = 200;
        r.set_header("Content-Type","application/json");
        r.write(W.dump());
        return r;
    });

    // 고도화된 GPT 생성 API
    CROW_ROUTE(app, "/api/generate").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req){
        initialize_model();
        
        auto body = crow::json::load(req.body);
        if(!body){
            return crow::response(400, "invalid json");
        }
        
        std::string prompt = body.has("prompt") ? body["prompt"].s() : std::string("hello world");
        int max_tokens = body.has("max_tokens") ? body["max_tokens"].i() : 10;
        std::string strategy = body.has("strategy") ? body["strategy"].s() : std::string("greedy");
        float temperature = body.has("temperature") ? body["temperature"].d() : 1.0f;
        int top_k = body.has("top_k") ? body["top_k"].i() : 50;
        float top_p = body.has("top_p") ? body["top_p"].d() : 0.9f;
        
        try {
            // 토큰화
            std::vector<int> prompt_tokens = global_tokenizer.encode(prompt);
            
            // 샘플링 설정
            SamplingConfig config;
            if (strategy == "top_k") {
                config.strategy = SamplingStrategy::TOP_K;
                config.top_k = top_k;
            } else if (strategy == "top_p") {
                config.strategy = SamplingStrategy::TOP_P;
                config.top_p = top_p;
            } else if (strategy == "temperature") {
                config.strategy = SamplingStrategy::TEMPERATURE;
                config.temperature = temperature;
            } else {
                config.strategy = SamplingStrategy::GREEDY;
            }
            
            // 생성
            auto start_time = std::chrono::high_resolution_clock::now();
            std::vector<int> generated_tokens = global_model->generate(prompt_tokens, max_tokens, config);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            // 디코딩
            std::string generated_text = global_tokenizer.decode(generated_tokens);
            
            // 성능 지표
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            const auto& metrics = global_model->get_performance_metrics();
            
            crow::json::wvalue response;
            response["prompt"] = prompt;
            response["generated"] = generated_text;
            response["prompt_tokens"] = prompt_tokens.size();
            response["generated_tokens"] = generated_tokens.size();
            response["total_time_ms"] = duration.count();
            response["strategy"] = strategy;
            response["config"]["temperature"] = temperature;
            response["config"]["top_k"] = top_k;
            response["config"]["top_p"] = top_p;
            
            // 성능 지표 추가
            crow::json::wvalue perf;
            for (const auto& [key, value] : metrics) {
                perf[key] = value;
            }
            response["performance"] = std::move(perf);
            
            return crow::response(200, response.dump());
            
        } catch (const std::exception& e) {
            crow::json::wvalue error;
            error["error"] = e.what();
            return crow::response(500, error.dump());
        }
    });

    // Flash Attention 시연 API
    CROW_ROUTE(app, "/api/flash_attention").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req){
        initialize_model();
        
        auto body = crow::json::load(req.body);
        if(!body){
            return crow::response(400, "invalid json");
        }
        
        // 입력 텍스트
        std::string text = body.has("text") ? body["text"].s() : std::string("hello world from flash attention");
        bool use_cache = body.has("use_cache") ? body["use_cache"].b() : true;
        
        try {
            // 토큰화
            std::vector<int> tokens = global_tokenizer.encode(text);
            if (tokens.empty()) {
                tokens = {2, 3, 4}; // 기본 토큰들
            }
            
            // 순전파 실행
            auto start_time = std::chrono::high_resolution_clock::now();
            AdvancedTensor logits = global_model->forward(tokens, use_cache);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            // 결과 구성
            crow::json::wvalue response;
            response["input_text"] = text;
            response["input_tokens"] = tokens;
            response["output_shape"] = std::vector<int>{logits.size(0), logits.size(1)};
            response["use_cache"] = use_cache;
            response["inference_time_us"] = duration.count();
            
            // 성능 지표
            const auto& metrics = global_model->get_performance_metrics();
            crow::json::wvalue perf;
            for (const auto& [key, value] : metrics) {
                perf[key] = value;
            }
            response["performance"] = std::move(perf);
            
            // 일부 로짓 샘플링 (처음 10개)
            std::vector<float> sample_logits;
            int sample_size = std::min(10, logits.size(1));
            for (int i = 0; i < sample_size; ++i) {
                sample_logits.push_back(logits(logits.size(0) - 1, i)); // 마지막 토큰의 로짓
            }
            response["sample_logits"] = sample_logits;
            
            return crow::response(200, response.dump());
            
        } catch (const std::exception& e) {
            crow::json::wvalue error;
            error["error"] = e.what();
            return crow::response(500, error.dump());
        }
    });

    const uint16_t port = 18080;
    std::cout << "Starting Advanced GPT server on port " << port << std::endl;
    app.port(port).multithreaded().run();
    return 0;
}
