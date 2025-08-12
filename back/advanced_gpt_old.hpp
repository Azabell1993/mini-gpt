#pragma once
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>

namespace gpt {

// 고도화된 Tensor 클래스
class AdvancedTensor {
private:
    std::vector<float> data_;
    std::vector<int> shape_;
    std::string device_;
    bool requires_grad_;

public:
    AdvancedTensor() = default;
    AdvancedTensor(const std::vector<int>& shape, const std::string& device = "cpu", bool requires_grad = false)
        : shape_(shape), device_(device), requires_grad_(requires_grad) {
        int total_size = 1;
        for (int dim : shape_) total_size *= dim;
        data_.resize(total_size, 0.0f);
    }

    // 접근자
    float& at(const std::vector<int>& indices) {
        int flat_idx = compute_flat_index(indices);
        return data_[flat_idx];
    }

    const float& at(const std::vector<int>& indices) const {
        int flat_idx = compute_flat_index(indices);
        return data_[flat_idx];
    }

    // 2D 접근 (편의용)
    float& operator()(int i, int j) {
        return data_[i * shape_[1] + j];
    }

    const float& operator()(int i, int j) const {
        return data_[i * shape_[1] + j];
    }

    // 속성
    const std::vector<int>& shape() const { return shape_; }
    int size(int dim) const { return shape_[dim]; }
    int numel() const { 
        int total = 1;
        for (int dim : shape_) total *= dim;
        return total;
    }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

    // 초기화 메서드
    void normal_(float mean = 0.0f, float std = 0.02f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> d(mean, std);
        for (auto& val : data_) {
            val = d(gen);
        }
    }

    void zeros_() {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }

    void ones_() {
        std::fill(data_.begin(), data_.end(), 1.0f);
    }

    // 연산 메서드
    AdvancedTensor matmul(const AdvancedTensor& other) const;
    AdvancedTensor transpose(int dim0, int dim1) const;
    AdvancedTensor view(const std::vector<int>& new_shape) const;
    void add_(const AdvancedTensor& other);
    void mul_(float scalar);
    void div_(float scalar);

private:
    int compute_flat_index(const std::vector<int>& indices) const {
        int flat_idx = 0;
        int stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            flat_idx += indices[i] * stride;
            stride *= shape_[i];
        }
        return flat_idx;
    }
};

// 고도화된 어텐션 메커니즘
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

    AdvancedTensor forward(const AdvancedTensor& x, bool is_causal = true, bool is_inference = false) const {
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
    AdvancedTensor forward_with_cache(const AdvancedTensor& Q, const AdvancedTensor& K, const AdvancedTensor& V, bool is_causal) const {
        // KV 캐시에 새로운 K, V 추가
        k_cache_.push_back(K);
        v_cache_.push_back(V);
        cache_length_++;

        // 전체 K, V 텐서 구성
        AdvancedTensor K_full({cache_length_, d_model_});
        AdvancedTensor V_full({cache_length_, d_model_});

        for (int i = 0; i < cache_length_; ++i) {
            for (int j = 0; j < d_model_; ++j) {
                K_full(i, j) = k_cache_[i](0, j);
                V_full(i, j) = v_cache_[i](0, j);
            }
        }

        return compute_attention(Q, K_full, V_full, is_causal);
    }

    AdvancedTensor forward_full(const AdvancedTensor& Q, const AdvancedTensor& K, const AdvancedTensor& V, bool is_causal) const {
        return compute_attention(Q, K, V, is_causal);
    }

    AdvancedTensor compute_attention(const AdvancedTensor& Q, const AdvancedTensor& K, const AdvancedTensor& V, bool is_causal) const {
        int seq_len_q = Q.size(0);
        int seq_len_kv = K.size(0);

        AdvancedTensor output({seq_len_q, d_model_});
        output.zeros_();

        // 헤드별 처리
        for (int h = 0; h < n_heads_; ++h) {
            // 헤드별 Q, K, V 추출
            AdvancedTensor Qh = extract_head(Q, h);
            AdvancedTensor Kh = extract_head(K, h);
            AdvancedTensor Vh = extract_head(V, h);

            // 스케일된 닷-프로덕트 어텐션
            AdvancedTensor head_out = scaled_dot_product_attention(Qh, Kh, Vh, is_causal);
            
            // 출력에 헤드 결과 병합
            merge_head(output, head_out, h);
        }

        // 출력 투영
        return output.matmul(Wo);
    }

    AdvancedTensor extract_head(const AdvancedTensor& tensor, int head_idx) const {
        int seq_len = tensor.size(0);
        AdvancedTensor head_tensor({seq_len, d_head_});
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_head_; ++j) {
                head_tensor(i, j) = tensor(i, head_idx * d_head_ + j);
            }
        }
        return head_tensor;
    }

    void merge_head(AdvancedTensor& output, const AdvancedTensor& head_out, int head_idx) const {
        int seq_len = head_out.size(0);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_head_; ++j) {
                output(i, head_idx * d_head_ + j) = head_out(i, j);
            }
        }
    }

    AdvancedTensor scaled_dot_product_attention(const AdvancedTensor& Q, const AdvancedTensor& K, const AdvancedTensor& V, bool is_causal) const {
        int seq_len_q = Q.size(0);
        int seq_len_kv = K.size(0);

        // scores = Q @ K^T / sqrt(d_head)
        AdvancedTensor scores({seq_len_q, seq_len_kv});
        for (int i = 0; i < seq_len_q; ++i) {
            for (int j = 0; j < seq_len_kv; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < d_head_; ++k) {
                    dot += Q(i, k) * K(j, k);
                }
                scores(i, j) = dot * scale_;
                
                // 인과적 마스킹
                if (is_causal && j > i) {
                    scores(i, j) = -1e9f;
                }
            }
        }

        // Softmax (행별)
        for (int i = 0; i < seq_len_q; ++i) {
            float max_val = -1e9f;
            for (int j = 0; j < seq_len_kv; ++j) {
                max_val = std::max(max_val, scores(i, j));
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len_kv; ++j) {
                scores(i, j) = std::exp(scores(i, j) - max_val);
                sum_exp += scores(i, j);
            }
            
            for (int j = 0; j < seq_len_kv; ++j) {
                scores(i, j) /= sum_exp;
            }
        }

        // output = scores @ V
        AdvancedTensor output({seq_len_q, d_head_});
        for (int i = 0; i < seq_len_q; ++i) {
            for (int j = 0; j < d_head_; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < seq_len_kv; ++k) {
                    sum += scores(i, k) * V(k, j);
                }
                output(i, j) = sum;
            }
        }

        return output;
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
                output(i, j) = (x(i, j) - mean) / std_dev * weight.data()[j] + bias.data()[j];
            }
        }

        return output;
    }
};

// 샘플링 전략
enum class SamplingStrategy {
    GREEDY,
    TOP_K,
    TOP_P,
    TEMPERATURE
};

struct SamplingConfig {
    SamplingStrategy strategy = SamplingStrategy::GREEDY;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float repetition_penalty = 1.1f;
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

public:
    AdvancedTensor token_embedding;
    AdvancedTensor position_embedding;
    std::vector<std::unique_ptr<FlashAttention>> attentions;
    std::vector<std::unique_ptr<AdvancedFFN>> ffns;
    std::vector<std::unique_ptr<LayerNorm>> ln1s;
    std::vector<std::unique_ptr<LayerNorm>> ln2s;
    std::unique_ptr<LayerNorm> ln_f;
    AdvancedTensor lm_head;

    // 성능 지표
    mutable std::unordered_map<std::string, double> performance_metrics;

    AdvancedGPT(int vocab_size, int d_model, int n_layers, int n_heads, int d_ff, int max_seq_len)
        : vocab_size_(vocab_size), d_model_(d_model), n_layers_(n_layers), 
          n_heads_(n_heads), d_ff_(d_ff), max_seq_len_(max_seq_len),
          token_embedding({vocab_size, d_model}),
          position_embedding({max_seq_len, d_model}),
          lm_head({d_model, vocab_size}) {

        // 임베딩 초기화
        token_embedding.normal_(0.0f, 0.02f);
        position_embedding.normal_(0.0f, 0.02f);
        lm_head.normal_(0.0f, 0.02f);

        // 레이어 초기화
        for (int i = 0; i < n_layers_; ++i) {
            attentions.push_back(std::make_unique<FlashAttention>(d_model_, n_heads_));
            ffns.push_back(std::make_unique<AdvancedFFN>(d_model_, d_ff_));
            ln1s.push_back(std::make_unique<LayerNorm>(d_model_));
            ln2s.push_back(std::make_unique<LayerNorm>(d_model_));
        }
        ln_f = std::make_unique<LayerNorm>(d_model_);
    }

    AdvancedTensor forward(const std::vector<int>& token_ids, bool is_inference = false) const {
        auto start_time = std::chrono::high_resolution_clock::now();

        int seq_len = token_ids.size();
        AdvancedTensor x({seq_len, d_model_});

        // 임베딩
        for (int i = 0; i < seq_len; ++i) {
            int token_id = token_ids[i] % vocab_size_;
            int pos_id = i % max_seq_len_;
            
            for (int j = 0; j < d_model_; ++j) {
                x(i, j) = token_embedding(token_id, j) + position_embedding(pos_id, j);
            }
        }

        // 트랜스포머 블록들
        for (int layer = 0; layer < n_layers_; ++layer) {
            auto layer_start = std::chrono::high_resolution_clock::now();

            // Pre-LayerNorm + Attention + Residual
            AdvancedTensor ln1_out = ln1s[layer]->forward(x);
            AdvancedTensor attn_out = attentions[layer]->forward(ln1_out, true, is_inference);
            x.add_(attn_out);  // Residual connection

            // Pre-LayerNorm + FFN + Residual
            AdvancedTensor ln2_out = ln2s[layer]->forward(x);
            AdvancedTensor ffn_out = ffns[layer]->forward(ln2_out);
            x.add_(ffn_out);   // Residual connection

            auto layer_end = std::chrono::high_resolution_clock::now();
            auto layer_duration = std::chrono::duration_cast<std::chrono::microseconds>(layer_end - layer_start);
            performance_metrics["layer_" + std::to_string(layer) + "_time_us"] = layer_duration.count();
        }

        // 최종 LayerNorm
        x = ln_f->forward(x);

        // 언어 모델 헤드
        AdvancedTensor logits = x.matmul(lm_head);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        performance_metrics["total_forward_time_us"] = total_duration.count();
        performance_metrics["tokens_per_second"] = seq_len * 1000000.0 / total_duration.count();

        return logits;
    }

    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens, const SamplingConfig& config = SamplingConfig{}) const {
        std::vector<int> generated = prompt;

        // KV 캐시 활성화
        for (auto& attn : attentions) {
            attn->enable_cache(max_seq_len_);
        }

        for (int step = 0; step < max_new_tokens; ++step) {
            // 마지막 토큰만 처리 (KV 캐시 활용)
            std::vector<int> current_token = {generated.back()};
            AdvancedTensor logits = forward(current_token, true);

            // 마지막 토큰의 로짓
            std::vector<float> last_logits(vocab_size_);
            for (int i = 0; i < vocab_size_; ++i) {
                last_logits[i] = logits(0, i);  // 배치 크기 1이므로 첫 번째 행
            }

            // 샘플링
            int next_token = sample(last_logits, config);
            generated.push_back(next_token);

            // EOS 토큰이면 종료
            if (next_token == 1) break;  // 1을 EOS로 가정
        }

        // 캐시 정리
        for (auto& attn : attentions) {
            attn->disable_cache();
        }

        return generated;
    }

    const std::unordered_map<std::string, double>& get_performance_metrics() const {
        return performance_metrics;
    }

private:
    int sample(const std::vector<float>& logits, const SamplingConfig& config) const {
        std::vector<float> probs = logits;

        // 온도 적용
        if (config.temperature != 1.0f) {
            for (auto& p : probs) {
                p /= config.temperature;
            }
        }

        // Softmax
        float max_logit = *std::max_element(probs.begin(), probs.end());
        float sum_exp = 0.0f;
        for (auto& p : probs) {
            p = std::exp(p - max_logit);
            sum_exp += p;
        }
        for (auto& p : probs) {
            p /= sum_exp;
        }

        switch (config.strategy) {
            case SamplingStrategy::GREEDY:
                return std::max_element(probs.begin(), probs.end()) - probs.begin();
            
            case SamplingStrategy::TOP_K:
                return sample_top_k(probs, config.top_k);
            
            case SamplingStrategy::TOP_P:
                return sample_top_p(probs, config.top_p);
            
            default:
                return sample_multinomial(probs);
        }
    }

    int sample_top_k(std::vector<float>& probs, int k) const {
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < probs.size(); ++i) {
            prob_idx.emplace_back(probs[i], i);
        }
        
        std::sort(prob_idx.rbegin(), prob_idx.rend());
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        float sum = 0.0f;
        for (int i = 0; i < std::min(k, (int)prob_idx.size()); ++i) {
            probs[prob_idx[i].second] = prob_idx[i].first;
            sum += prob_idx[i].first;
        }
        
        for (auto& p : probs) p /= sum;
        
        return sample_multinomial(probs);
    }

    int sample_top_p(std::vector<float>& probs, float p) const {
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < probs.size(); ++i) {
            prob_idx.emplace_back(probs[i], i);
        }
        
        std::sort(prob_idx.rbegin(), prob_idx.rend());
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        float cumsum = 0.0f;
        for (const auto& [prob, idx] : prob_idx) {
            probs[idx] = prob;
            cumsum += prob;
            if (cumsum >= p) break;
        }
        
        for (auto& prob : probs) prob /= cumsum;
        
        return sample_multinomial(probs);
    }

    int sample_multinomial(const std::vector<float>& probs) const {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(gen);
    }
};

} // namespace gpt
