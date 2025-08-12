#pragma once

// 통합된 Advanced GPT 구현
// mini-transformer 프로젝트의 고도화된 구현을 mini-gpt-paper-skeleton에 통합

#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace gpt {

// 기본 텐서 클래스 (advanced_tensor.cpp와 호환)
class AdvancedTensor {
private:
    std::vector<float> data_;
    std::vector<int> shape_;
    std::string device_;
    bool requires_grad_;

public:
    AdvancedTensor() : device_("cpu"), requires_grad_(false) {}
    AdvancedTensor(const std::vector<int>& shape, const std::string& device = "cpu", bool requires_grad = false)
        : shape_(shape), device_(device), requires_grad_(requires_grad) {
        int total_size = 1;
        for (int dim : shape) total_size *= dim;
        data_.resize(total_size, 0.0f);
    }

    // 기본 접근자
    int size(int dim) const { return (dim < shape_.size()) ? shape_[dim] : 1; }
    float& operator()(int i, int j) { return data_[i * shape_[1] + j]; }
    const float& operator()(int i, int j) const { return data_[i * shape_[1] + j]; }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

    // 텐서 연산 (advanced_tensor.cpp에서 구현)
    AdvancedTensor matmul(const AdvancedTensor& other) const;
    AdvancedTensor transpose(int dim0 = 0, int dim1 = 1) const;
    
    // 유틸리티 함수들
    void zeros_() { std::fill(data_.begin(), data_.end(), 0.0f); }
    void ones_() { std::fill(data_.begin(), data_.end(), 1.0f); }
    void normal_(float mean = 0.0f, float std = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(mean, std);
        for (auto& x : data_) x = dis(gen);
    }
};

// GELU 활성화 함수
class GELU {
public:
    static AdvancedTensor forward(const AdvancedTensor& x) {
        AdvancedTensor output = x;
        auto& data = output.data();
        
        for (auto& val : data) {
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            float x3 = val * val * val;
            float tanh_arg = std::sqrt(2.0f / M_PI) * (val + 0.044715f * x3);
            val = 0.5f * val * (1.0f + std::tanh(tanh_arg));
        }
        
        return output;
    }
};

// LayerNorm
class LayerNorm {
private:
    int d_model_;
    float eps_;

public:
    AdvancedTensor weight, bias;

    LayerNorm(int d_model, float eps = 1e-5f) 
        : d_model_(d_model), eps_(eps),
          weight({d_model}), bias({d_model}) {
        weight.ones_();
        bias.zeros_();
    }

    AdvancedTensor forward(const AdvancedTensor& x) const {
        int seq_len = x.size(0);
        AdvancedTensor output = x;

        for (int i = 0; i < seq_len; ++i) {
            // 평균 계산
            float mean = 0.0f;
            for (int j = 0; j < d_model_; ++j) {
                mean += x(i, j);
            }
            mean /= d_model_;

            // 분산 계산
            float var = 0.0f;
            for (int j = 0; j < d_model_; ++j) {
                float diff = x(i, j) - mean;
                var += diff * diff;
            }
            var /= d_model_;

            // 정규화
            float std_dev = std::sqrt(var + eps_);
            for (int j = 0; j < d_model_; ++j) {
                output(i, j) = (x(i, j) - mean) / std_dev 
                             * weight.data()[j] + bias.data()[j];
            }
        }

        return output;
    }
};

// Flash Attention 구현
class FlashAttention {
private:
    int d_model_;
    int n_heads_;
    int d_head_;
    float scale_;
    
    // KV 캐시
    mutable std::vector<AdvancedTensor> k_cache_;
    mutable std::vector<AdvancedTensor> v_cache_;
    mutable bool use_cache_;
    mutable int cache_length_;

public:
    AdvancedTensor Wq, Wk, Wv, Wo;

    FlashAttention(int d_model, int n_heads) 
        : d_model_(d_model), n_heads_(n_heads), d_head_(d_model / n_heads),
          scale_(1.0f / std::sqrt(float(d_head_))), use_cache_(false), cache_length_(0),
          Wq({d_model, d_model}), Wk({d_model, d_model}), 
          Wv({d_model, d_model}), Wo({d_model, d_model}) {
        
        // Xavier 초기화
        float std_dev = std::sqrt(2.0f / (d_model + d_model));
        Wq.normal_(0.0f, std_dev);
        Wk.normal_(0.0f, std_dev);
        Wv.normal_(0.0f, std_dev);
        Wo.normal_(0.0f, std_dev);
    }

    void enable_cache(int max_seq_len) {
        use_cache_ = true;
        cache_length_ = 0;
        k_cache_.clear();
        v_cache_.clear();
        k_cache_.reserve(max_seq_len);
        v_cache_.reserve(max_seq_len);
    }

    void disable_cache() {
        use_cache_ = false;
        cache_length_ = 0;
        k_cache_.clear();
        v_cache_.clear();
    }

    AdvancedTensor forward(const AdvancedTensor& x, bool is_causal = true, 
                          bool is_inference = false) const {
        int seq_len = x.size(0);
        int d_model = x.size(1);

        // Q, K, V 계산
        AdvancedTensor Q = x.matmul(Wq);
        AdvancedTensor K = x.matmul(Wk);
        AdvancedTensor V = x.matmul(Wv);

        if (is_inference && use_cache_ && seq_len == 1) {
            return forward_with_cache(Q, K, V, is_causal);
        } else {
            return forward_full(Q, K, V, is_causal);
        }
    }

private:
    AdvancedTensor forward_full(const AdvancedTensor& Q, const AdvancedTensor& K, 
                               const AdvancedTensor& V, bool is_causal) const {
        int seq_len = Q.size(0);
        
        // 스코어 계산 Q @ K^T
        AdvancedTensor scores = Q.matmul(K.transpose());
        
        // 스케일링
        for (auto& val : scores.data()) {
            val *= scale_;
        }
        
        // Causal 마스킹
        if (is_causal) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = i + 1; j < seq_len; ++j) {
                    scores(i, j) = -1e9f;
                }
            }
        }
        
        // Softmax (간단한 구현)
        AdvancedTensor attn_weights = scores;
        for (int i = 0; i < seq_len; ++i) {
            float max_val = attn_weights(i, 0);
            for (int j = 1; j < seq_len; ++j) {
                max_val = std::max(max_val, attn_weights(i, j));
            }
            
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                attn_weights(i, j) = std::exp(attn_weights(i, j) - max_val);
                sum += attn_weights(i, j);
            }
            
            for (int j = 0; j < seq_len; ++j) {
                attn_weights(i, j) /= sum;
            }
        }
        
        // 출력 계산
        AdvancedTensor output = attn_weights.matmul(V);
        return output.matmul(Wo);
    }
    
    AdvancedTensor forward_with_cache(const AdvancedTensor& Q, const AdvancedTensor& K, 
                                     const AdvancedTensor& V, bool is_causal) const {
        // KV 캐시 구현 (간단화)
        return forward_full(Q, K, V, is_causal);
    }
};

// 고도화된 FFN
class AdvancedFFN {
private:
    int d_model_;
    int d_ff_;

public:
    AdvancedTensor W1, W2, b1, b2;

    AdvancedFFN(int d_model, int d_ff) 
        : d_model_(d_model), d_ff_(d_ff),
          W1({d_model, d_ff}), W2({d_ff, d_model}),
          b1({1, d_ff}), b2({1, d_model}) {
        
        // Xavier 초기화
        float std_dev1 = std::sqrt(2.0f / (d_model + d_ff));
        float std_dev2 = std::sqrt(2.0f / (d_ff + d_model));
        
        W1.normal_(0.0f, std_dev1);
        W2.normal_(0.0f, std_dev2);
        b1.zeros_();
        b2.zeros_();
    }

    AdvancedTensor forward(const AdvancedTensor& x) const {
        // x @ W1 + b1
        AdvancedTensor h1 = x.matmul(W1);
        
        // bias 추가
        for (int i = 0; i < h1.size(0); ++i) {
            for (int j = 0; j < h1.size(1); ++j) {
                h1(i, j) += b1(0, j);
            }
        }

        // GELU 활성화
        h1 = GELU::forward(h1);

        // h1 @ W2 + b2
        AdvancedTensor output = h1.matmul(W2);
        
        // bias 추가
        for (int i = 0; i < output.size(0); ++i) {
            for (int j = 0; j < output.size(1); ++j) {
                output(i, j) += b2(0, j);
            }
        }

        return output;
    }
};

// 트랜스포머 블록
class TransformerBlock {
public:
    LayerNorm ln1, ln2;
    FlashAttention attn;
    AdvancedFFN ffn;

    TransformerBlock(int d_model, int n_heads, int d_ff) 
        : ln1(d_model), ln2(d_model), attn(d_model, n_heads), ffn(d_model, d_ff) {}

    AdvancedTensor forward(const AdvancedTensor& x, bool is_inference = false) const {
        // Pre-LN: LayerNorm → Attention → Residual
        AdvancedTensor norm1 = ln1.forward(x);
        AdvancedTensor attn_out = attn.forward(norm1, true, is_inference);
        
        // Residual connection
        AdvancedTensor x1 = x;
        for (int i = 0; i < x1.size(0); ++i) {
            for (int j = 0; j < x1.size(1); ++j) {
                x1(i, j) += attn_out(i, j);
            }
        }
        
        // Pre-LN: LayerNorm → FFN → Residual
        AdvancedTensor norm2 = ln2.forward(x1);
        AdvancedTensor ffn_out = ffn.forward(norm2);
        
        // Residual connection
        for (int i = 0; i < x1.size(0); ++i) {
            for (int j = 0; j < x1.size(1); ++j) {
                x1(i, j) += ffn_out(i, j);
            }
        }
        
        return x1;
    }
};

// 샘플링 전략
enum class SamplingStrategy { GREEDY, TOP_K, TOP_P, TEMPERATURE };

struct SamplingConfig {
    SamplingStrategy strategy = SamplingStrategy::GREEDY;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
};

// 고도화된 GPT 모델
class AdvancedGPT {
private:
    int vocab_size_;
    int d_model_;
    int n_layers_;
    int n_heads_;
    int d_ff_;
    int max_seq_len_;
    
    std::vector<TransformerBlock> blocks_;
    AdvancedTensor token_embedding_;
    AdvancedTensor pos_embedding_;
    LayerNorm final_ln_;
    AdvancedTensor output_projection_;
    
    // 성능 지표
    mutable std::unordered_map<std::string, float> performance_metrics_;

public:
    AdvancedGPT(int vocab_size, int d_model, int n_layers, int n_heads, int d_ff, int max_seq_len)
        : vocab_size_(vocab_size), d_model_(d_model), n_layers_(n_layers), 
          n_heads_(n_heads), d_ff_(d_ff), max_seq_len_(max_seq_len),
          token_embedding_({vocab_size, d_model}), pos_embedding_({max_seq_len, d_model}),
          final_ln_(d_model), output_projection_({d_model, vocab_size}) {
        
        // 트랜스포머 블록들 초기화
        blocks_.reserve(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            blocks_.emplace_back(d_model, n_heads, d_ff);
        }
        
        // 가중치 초기화
        token_embedding_.normal_(0.0f, 0.02f);
        pos_embedding_.normal_(0.0f, 0.02f);
        output_projection_.normal_(0.0f, 0.02f);
    }

    AdvancedTensor forward(const std::vector<int>& tokens, bool use_cache = false) const {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int seq_len = tokens.size();
        
        // 토큰 임베딩 + 위치 임베딩
        AdvancedTensor x({seq_len, d_model_});
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_model_; ++j) {
                x(i, j) = token_embedding_(tokens[i], j) + pos_embedding_(i, j);
            }
        }
        
        // 트랜스포머 블록들
        for (int layer = 0; layer < n_layers_; ++layer) {
            auto layer_start = std::chrono::high_resolution_clock::now();
            x = blocks_[layer].forward(x, use_cache);
            auto layer_end = std::chrono::high_resolution_clock::now();
            
            auto layer_duration = std::chrono::duration_cast<std::chrono::microseconds>(layer_end - layer_start);
            performance_metrics_[std::string("layer_") + std::to_string(layer) + "_time_us"] = layer_duration.count();
        }
        
        // 최종 LayerNorm
        x = final_ln_.forward(x);
        
        // 출력 프로젝션
        AdvancedTensor logits = x.matmul(output_projection_);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        performance_metrics_["total_forward_time_us"] = total_duration.count();
        performance_metrics_["tokens_per_second"] = (seq_len * 1000000.0f) / total_duration.count();
        
        return logits;
    }

    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens, 
                             const SamplingConfig& config = SamplingConfig{}) const {
        std::vector<int> tokens = prompt;
        
        for (int i = 0; i < max_new_tokens; ++i) {
            AdvancedTensor logits = forward(tokens, true);
            
            // 마지막 토큰의 로짓 사용
            int last_pos = logits.size(0) - 1;
            std::vector<float> last_logits;
            for (int j = 0; j < vocab_size_; ++j) {
                last_logits.push_back(logits(last_pos, j));
            }
            
            int next_token = sample_token(last_logits, config);
            tokens.push_back(next_token);
            
            // EOS 토큰이면 종료
            if (next_token == 1) break;
        }
        
        return tokens;
    }

    const std::unordered_map<std::string, float>& get_performance_metrics() const {
        return performance_metrics_;
    }

private:
    int sample_token(const std::vector<float>& logits, const SamplingConfig& config) const {
        switch (config.strategy) {
            case SamplingStrategy::GREEDY:
                return std::max_element(logits.begin(), logits.end()) - logits.begin();
            
            case SamplingStrategy::TEMPERATURE: {
                std::vector<float> probs = logits;
                for (auto& p : probs) p /= config.temperature;
                return sample_from_probs(softmax(probs));
            }
            
            default:
                return std::max_element(logits.begin(), logits.end()) - logits.begin();
        }
    }
    
    std::vector<float> softmax(const std::vector<float>& x) const {
        std::vector<float> y(x.size());
        float max_val = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            y[i] = std::exp(x[i] - max_val);
            sum += y[i];
        }
        
        for (auto& val : y) val /= sum;
        return y;
    }
    
    int sample_from_probs(const std::vector<float>& probs) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        float r = dis(gen);
        float cumsum = 0.0f;
        
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r <= cumsum) return i;
        }
        
        return probs.size() - 1;
    }
};

} // namespace gpt
