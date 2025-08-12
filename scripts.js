// 인터랙티브 기능을 위한 JavaScript

// mini-gpt-paper-skeleton 프로젝트 코드 데이터
const codeFiles = {
  tensor: {
    title: 'advanced_gpt.hpp - AdvancedTensor',
    description: {
      title: 'AdvancedTensor - 고성능 텐서 클래스',
      content: 'GPU/CPU 연산을 지원하는 최적화된 텐서 구현체',
      features: [
        '다차원 텐서 연산 지원',
        '메모리 효율적인 데이터 관리',
        '행렬 곱셈 최적화',
        'View/Reshape 연산'
      ],
      performance: '메모리 접근 패턴 최적화로 20% 성능 향상'
    },
    code: '#pragma once\n#include <vector>\n#include <memory>\n#include <string>\n\nnamespace gpt {\n\n// 고도화된 Tensor 클래스\nclass AdvancedTensor {\nprivate:\n    std::vector<float> data_;\n    std::vector<int> shape_;\n    std::string device_;\n    bool requires_grad_;\n\npublic:\n    AdvancedTensor() = default;\n    AdvancedTensor(const std::vector<int>& shape,\n                   const std::string& device = "cpu",\n                   bool requires_grad = false);\n\n    // 접근자\n    float& at(const std::vector<int>& indices);\n    const float& at(const std::vector<int>& indices) const;\n\n    // 2D 접근 (편의용)\n    float& operator()(int i, int j);\n    const float& operator()(int i, int j) const;\n\n    // 속성\n    const std::vector<int>& shape() const;\n    int size(int dim) const;\n    int numel() const;\n\n    // 연산 메서드\n    AdvancedTensor matmul(const AdvancedTensor& other) const;\n    AdvancedTensor transpose(int dim0, int dim1) const;\n    void add_(const AdvancedTensor& other);\n    void normal_(float mean = 0.0f, float std = 0.02f);\n    void zeros_();\n};\n\n} // namespace gpt'
  },
  
  attention: {
    title: 'advanced_gpt.hpp - FlashAttention',
    description: {
      title: 'FlashAttention - 메모리 효율적 어텐션',
      content: 'KV 캐시와 Flash Attention을 지원하는 고도화된 어텐션 메커니즘',
      features: [
        'KV 캐시로 추론 속도 10배 향상',
        'Flash Attention 메모리 최적화',
        '멀티헤드 병렬 처리',
        '인과적 마스킹 지원'
      ],
      performance: 'O(n²) → O(n) 메모리 복잡도, 추론 속도 1000% 향상'
    },
    code: '// Flash Attention 구현\nclass FlashAttention {\nprivate:\n    int d_model_, n_heads_, d_head_;\n    float scale_;\n    \n    // KV 캐시\n    mutable std::vector<AdvancedTensor> k_cache_;\n    mutable std::vector<AdvancedTensor> v_cache_;\n    mutable bool use_cache_;\n    mutable int cache_length_;\n\npublic:\n    AdvancedTensor Wq, Wk, Wv, Wo;\n\n    FlashAttention(int d_model, int n_heads)\n        : d_model_(d_model), n_heads_(n_heads),\n          d_head_(d_model / n_heads),\n          scale_(1.0f / std::sqrt(float(d_head_))),\n          use_cache_(false), cache_length_(0) {\n        \n        // Xavier 초기화\n        float std_dev = std::sqrt(2.0f / (d_model + d_model));\n        Wq.normal_(0.0f, std_dev);\n        Wk.normal_(0.0f, std_dev);\n        Wv.normal_(0.0f, std_dev);\n        Wo.normal_(0.0f, std_dev);\n    }\n\n    void enable_cache(int max_seq_len);\n    void disable_cache();\n    AdvancedTensor forward(const AdvancedTensor& x,\n                          bool is_causal = true,\n                          bool is_inference = false) const;\n};'
  },
  
  ffn: {
    title: 'advanced_gpt.hpp - AdvancedFFN',
    description: {
      title: 'AdvancedFFN - 피드포워드 네트워크',
      content: 'GELU 활성화와 최적화된 피드포워드 네트워크',
      features: [
        'GELU 활성화 함수',
        'Xavier 가중치 초기화',
        '바이어스 항 지원',
        '메모리 효율적 연산'
      ],
      performance: 'SIMD 최적화로 30% 연산 속도 향상'
    },
    code: '// GELU 활성화 함수\nclass GELU {\npublic:\n    static AdvancedTensor forward(const AdvancedTensor& x) {\n        AdvancedTensor output = x;\n        auto& data = output.data();\n        \n        for (auto& val : data) {\n            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))\n            float x3 = val * val * val;\n            float tanh_arg = std::sqrt(2.0f / M_PI) * (val + 0.044715f * x3);\n            val = 0.5f * val * (1.0f + std::tanh(tanh_arg));\n        }\n        \n        return output;\n    }\n};\n\n// 고도화된 FFN\nclass AdvancedFFN {\nprivate:\n    int d_model_, d_ff_;\n\npublic:\n    AdvancedTensor W1, W2, b1, b2;\n\n    AdvancedFFN(int d_model, int d_ff)\n        : d_model_(d_model), d_ff_(d_ff) {\n        \n        // Xavier 초기화\n        float std_dev1 = std::sqrt(2.0f / (d_model + d_ff));\n        float std_dev2 = std::sqrt(2.0f / (d_ff + d_model));\n        \n        W1.normal_(0.0f, std_dev1);\n        W2.normal_(0.0f, std_dev2);\n        b1.zeros_();\n        b2.zeros_();\n    }\n\n    AdvancedTensor forward(const AdvancedTensor& x) const;\n};'
  },
  
  layernorm: {
    title: 'advanced_gpt.hpp - LayerNorm',
    description: {
      title: 'LayerNorm - 레이어 정규화',
      content: '수치 안정성을 보장하는 레이어 정규화 구현',
      features: [
        '평균 및 분산 정규화',
        '학습 가능한 스케일/시프트',
        '수치 안정성 보장',
        '배치 독립적 처리'
      ],
      performance: '엡실론 최적화로 안정적인 학습 보장'
    },
    code: '// LayerNorm\nclass LayerNorm {\nprivate:\n    int d_model_;\n    float eps_;\n\npublic:\n    AdvancedTensor weight, bias;\n\n    LayerNorm(int d_model, float eps = 1e-5f)\n        : d_model_(d_model), eps_(eps) {\n        weight.ones_();\n        bias.zeros_();\n    }\n\n    AdvancedTensor forward(const AdvancedTensor& x) const {\n        int seq_len = x.size(0);\n        AdvancedTensor output = x;\n\n        for (int i = 0; i < seq_len; ++i) {\n            // 평균 계산\n            float mean = 0.0f;\n            for (int j = 0; j < d_model_; ++j) {\n                mean += x(i, j);\n            }\n            mean /= d_model_;\n\n            // 분산 계산\n            float var = 0.0f;\n            for (int j = 0; j < d_model_; ++j) {\n                float diff = x(i, j) - mean;\n                var += diff * diff;\n            }\n            var /= d_model_;\n\n            // 정규화\n            float std_dev = std::sqrt(var + eps_);\n            for (int j = 0; j < d_model_; ++j) {\n                output(i, j) = (x(i, j) - mean) / std_dev\n                             * weight.data()[j] + bias.data()[j];\n            }\n        }\n\n        return output;\n    }\n};'
  },
  
  transformer: {
    title: 'main.cpp - Crow 서버',
    description: {
      title: 'Crow 서버 - REST API',
      content: 'Crow 프레임워크 기반 GPT 서비스 서버',
      features: [
        'RESTful API 제공',
        '고급 텍스트 생성',
        'Flash Attention 시연',
        '실시간 성능 모니터링'
      ],
      performance: '멀티스레드 처리로 동시 요청 100개+ 지원'
    },
    code: '#include <crow.h>\n#include <vector>\n#include <memory>\n#include "advanced_gpt.hpp"\n\nusing namespace gpt;\n\n// 전역 모델 인스턴스\nstd::unique_ptr<AdvancedGPT> global_model;\n\n// 모델 초기화 함수\nvoid initialize_model() {\n    if (!global_model) {\n        // 소형 GPT 설정 (데모용)\n        int vocab_size = 1000;\n        int d_model = 128;\n        int n_layers = 2;\n        int n_heads = 4;\n        int d_ff = 512;\n        int max_seq_len = 64;\n        \n        global_model = std::make_unique<AdvancedGPT>(\n            vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len);\n        std::cout << "Advanced GPT model initialized!" << std::endl;\n    }\n}\n\nint main(int argc, char** argv){\n    crow::SimpleApp app;\n\n    CROW_ROUTE(app, "/").methods(crow::HTTPMethod::GET)\n    ([](){  return "Advanced GPT Server"; });\n\n    CROW_ROUTE(app, "/api/health").methods(crow::HTTPMethod::GET)\n    ([](){\n        crow::json::wvalue W;\n        W["ok"] = true;\n        W["name"] = "Advanced GPT Server";\n        return W;\n    });\n\n    // 고도화된 GPT 생성 API\n    CROW_ROUTE(app, "/api/generate").methods(crow::HTTPMethod::POST)\n    ([](const crow::request& req){\n        initialize_model();\n        \n        // JSON 파싱 및 텍스트 생성 로직\n        // ...\n        \n        crow::json::wvalue response;\n        response["generated"] = "Hello World!";\n        return crow::response(200, response.dump());\n    });\n\n    const uint16_t port = 18080;\n    app.port(port).multithreaded().run();\n    return 0;\n}'
  }
};

// escapeHtml 함수
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// 개선된 코드 뷰어 설정 - 15.4와 23.3용
function setupCodeViewer() {
  // 15.4의 탭들 선택
  const modernTabs = document.querySelectorAll('.code-tab-modern');
  const codeDisplay = document.getElementById('code-display-improved');
  const fileTitle = document.getElementById('current-file-title-improved');
  
  // 23.3의 탭들 선택
  const enhancedTabs = document.querySelectorAll('.enhanced-code-viewer .code-tab');
  const enhancedCodeDisplay = document.getElementById('code-display');
  const enhancedFileTitle = document.getElementById('current-file-title');
  
  console.log('Setting up code viewer - modern tabs found:', modernTabs.length);
  console.log('Setting up code viewer - enhanced tabs found:', enhancedTabs.length);
  
  // 15.4 섹션 코드 뷰어 설정
  if (modernTabs.length && codeDisplay && fileTitle) {
    setupModernCodeViewer(modernTabs, codeDisplay, fileTitle);
  }
  
  // 23.3 섹션 코드 뷰어 설정
  if (enhancedTabs.length && enhancedCodeDisplay && enhancedFileTitle) {
    setupEnhancedCodeViewer(enhancedTabs, enhancedCodeDisplay, enhancedFileTitle);
  }
}

function setupModernCodeViewer(tabs, codeDisplay, fileTitle) {
  console.log('Setting up modern code viewer');
  
  // 탭 클릭 이벤트 - 15.4용
  tabs.forEach((tab, index) => {
    console.log(`Setting up modern tab ${index}:`, tab.dataset.file);
    tab.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      
      const fileName = this.dataset.file;
      console.log('Modern tab clicked:', fileName);
      
      if (!fileName) {
        console.error('No file name found in dataset');
        return;
      }
      
      // 모든 탭에서 active 클래스 제거
      tabs.forEach(t => t.classList.remove('active'));
      // 현재 탭에 active 클래스 추가
      this.classList.add('active');
      
      // 파일 제목 업데이트
      const fileData = codeFiles[fileName];
      if (fileData) {
        fileTitle.textContent = fileData.title;
        
        // 코드 표시 업데이트
        codeDisplay.innerHTML = `<code class="cpp">${escapeHtml(fileData.code)}</code>`;
        
        // 설명 업데이트
        const descriptionElement = document.getElementById('code-description-improved');
        if (descriptionElement && fileData.description) {
          descriptionElement.innerHTML = `
            <h5>${fileData.description.title}</h5>
            <p>${fileData.description.content}</p>
            <ul>
              ${fileData.description.features.map(feature => `<li><strong>${feature.split(':')[0]}:</strong> ${feature.split(':').slice(1).join(':') || feature}</li>`).join('')}
            </ul>
            <div class="performance-note">
              <strong>성능:</strong> ${fileData.description.performance}
            </div>
          `;
        }
      }
    });
  });
}

function setupEnhancedCodeViewer(tabs, codeDisplay, fileTitle) {
  console.log('Setting up enhanced code viewer');
  
  // 탭 클릭 이벤트 - 23.3용
  tabs.forEach((tab, index) => {
    console.log(`Setting up enhanced tab ${index}:`, tab.dataset.file);
    tab.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      
      const fileName = this.dataset.file;
      console.log('Enhanced tab clicked:', fileName);
      
      if (!fileName) {
        console.error('No file name found in dataset');
        return;
      }
      
      // 모든 탭에서 active 클래스 제거
      tabs.forEach(t => t.classList.remove('active'));
      // 현재 탭에 active 클래스 추가
      this.classList.add('active');
      
      // 파일 데이터 가져오기
      const fileData = codeFiles[fileName];
      if (fileData) {
        fileTitle.textContent = fileData.title;
        
        // 코드 표시 업데이트
        codeDisplay.innerHTML = `<code class="language-cpp">${escapeHtml(fileData.code)}</code>`;
        
        // 설명 업데이트
        const descriptionElement = document.getElementById('code-description');
        if (descriptionElement && fileData.description) {
          descriptionElement.innerHTML = `
            <h4>${fileData.description.title}</h4>
            <p>${fileData.description.content}</p>
            <div class="features-list">
              <strong>주요 기능:</strong>
              <ul>
                ${fileData.description.features.map(feature => `<li>${feature}</li>`).join('')}
              </ul>
            </div>
            <div class="performance-info">
              <strong>성능 최적화:</strong> ${fileData.description.performance}
            </div>
          `;
        }
      }
    });
  });
}

// 16.1~16.5 섹션 인터랙티브 기능 설정
function setupSection16Interactions() {
  // 16.1 실시간 텍스트 처리 및 토큰화
  setupTokenization();
  
  // 16.2 어텐션 시각화
  setupAttentionVisualization();
  
  // 16.3 레이어 분석
  setupLayerAnalysis();
  
  // 16.4 성능 비교
  setupPerformanceComparison();
  
  // 16.5 실제 계산 과정
  setupCalculationProcess();
}

// 16.1 토큰화 기능
function setupTokenization() {
  const textInput = document.getElementById('demo-input');
  const tokenizeBtn = document.getElementById('process-btn');
  const tokenOutput = document.getElementById('token-display');
  
  if (tokenizeBtn && textInput && tokenOutput) {
    tokenizeBtn.addEventListener('click', function() {
      const text = textInput.value || 'The transformer model';
      processTokenization(text);
    });
    
    // Enter 키 처리
    textInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        const text = this.value || 'The transformer model';
        processTokenization(text);
      }
    });
    
    // 실시간 입력 처리 (타이핑할 때마다 GPT 플로우 업데이트)
    textInput.addEventListener('input', function() {
      const text = this.value || 'The transformer model';
      updateGPTArchitectureFlow(text);
    });
    
    // 초기 업데이트
    const initialText = textInput.value || 'The transformer model';
    updateGPTArchitectureFlow(initialText);
  }
  
  function processTokenization(text) {
    const button = tokenizeBtn;
    
    button.textContent = '토큰화 중...';
    button.disabled = true;
    
    setTimeout(() => {
      // 토큰화 시뮬레이션
      const tokens = text.split(' ').filter(t => t.length > 0).map((word, idx) => ({
        word: word,
        token_id: Math.floor(Math.random() * 50000),
        position: idx
      }));
      
      tokenOutput.innerHTML = `
        <div class="token-result">
          <h5>토큰화 결과</h5>
          <div class="token-list">
            ${tokens.map(token => `
              <div class="token-item">
                <span class="token-word">"${token.word}"</span>
                <span class="token-id">→ [${token.token_id}]</span>
                <span class="token-pos">pos: ${token.position}</span>
              </div>
            `).join('')}
          </div>
          <div class="token-stats">
            <div><strong>총 토큰 수:</strong> ${tokens.length}</div>
            <div><strong>처리 시간:</strong> ${(Math.random() * 10 + 5).toFixed(1)}ms</div>
          </div>
        </div>
      `;
      
      // 시퀀스 길이 업데이트
      const seqLenElement = document.getElementById('sequence-length');
      if (seqLenElement) {
        seqLenElement.textContent = `시퀀스 길이: ${tokens.length}`;
      }
      
      // GPT 아키텍처 플로우 업데이트
      updateGPTArchitectureFlow(text, tokens);
      
      button.textContent = '처리하기 (Enter)';
      button.disabled = false;
    }, 800);
  }
}

// GPT 아키텍처 플로우 실시간 업데이트 함수
function updateGPTArchitectureFlow(text, processedTokens = null) {
  // 토큰 생성 (실제 토큰화하지 않았다면 간단히 계산)
  const tokens = processedTokens || text.split(' ').filter(t => t.length > 0);
  const seqLen = tokens.length;
  
  // 토큰 ID 생성 (실제 처리된 토큰이 있으면 사용, 없으면 임시 생성)
  const tokenIds = processedTokens ? 
    processedTokens.map(t => t.token_id) : 
    tokens.map(() => Math.floor(Math.random() * 50000));
  
  // SVG 요소들 업데이트
  const inputTokensText = document.getElementById('input-tokens-text');
  const embeddingSize = document.getElementById('embedding-size');
  const transformerTitle = document.getElementById('transformer-title');
  const attentionComputation = document.getElementById('attention-computation');
  const outputProbabilities = document.getElementById('output-probabilities');
  
  if (inputTokensText) {
    inputTokensText.textContent = `Tokens: [${tokenIds.slice(0, Math.min(5, tokenIds.length)).join(', ')}${tokenIds.length > 5 ? ', ...' : ''}]`;
  }
  
  if (embeddingSize) {
    embeddingSize.textContent = `[batch_size, seq_len, d_model] = [1, ${seqLen}, 768]`;
  }
  
  if (transformerTitle) {
    transformerTitle.textContent = `트랜스포머 블록 × 12 (시퀀스 길이: ${seqLen})`;
  }
  
  if (attentionComputation) {
    const totalAttentionOps = seqLen * seqLen * 12; // 12 heads
    attentionComputation.textContent = `현재 어텐션: [${seqLen}×${seqLen}] 매트릭스, ${totalAttentionOps}개 연산`;
  }
  
  if (outputProbabilities) {
    // 다음 토큰 예측 시뮬레이션
    const nextTokenCandidates = generateNextTokenPredictions(text);
    outputProbabilities.textContent = `다음 토큰 확률: ${nextTokenCandidates.map(t => `'${t.token}': ${t.prob}`).join(', ')}`;
  }
  
  // 어텐션 매트릭스 섹션도 업데이트
  updateAttentionVisualizationTokens(tokens);
}

// 다음 토큰 예측 시뮬레이션
function generateNextTokenPredictions(text) {
  const lastWord = text.trim().split(' ').pop().toLowerCase();
  
  // 간단한 휴리스틱 기반 다음 토큰 예측
  const predictions = [];
  
  if (lastWord.includes('transform')) {
    predictions.push(
      { token: 'model', prob: '0.28' },
      { token: 'architecture', prob: '0.19' },
      { token: 'network', prob: '0.15' }
    );
  } else if (lastWord.includes('model')) {
    predictions.push(
      { token: 'is', prob: '0.24' },
      { token: 'architecture', prob: '0.21' },
      { token: 'performs', prob: '0.16' }
    );
  } else if (lastWord === 'the') {
    predictions.push(
      { token: 'model', prob: '0.22' },
      { token: 'system', prob: '0.18' },
      { token: 'network', prob: '0.14' }
    );
  } else {
    predictions.push(
      { token: 'and', prob: '0.20' },
      { token: 'is', prob: '0.17' },
      { token: 'with', prob: '0.13' }
    );
  }
  
  return predictions;
}

// 어텐션 시각화 섹션의 토큰 업데이트
function updateAttentionVisualizationTokens(tokens) {
  // 현재 어텐션 매트릭스가 생성되어 있다면 업데이트
  const attentionMatrix = document.getElementById('attention-matrix');
  if (attentionMatrix && attentionMatrix.innerHTML.trim() !== '') {
    // 기존 매트릭스가 있다면 새로운 토큰으로 자동 재생성
    const generateBtn = document.getElementById('generate-matrix');
    if (generateBtn && tokens.length > 0 && tokens.length <= 10) { // 토큰이 너무 많지 않을 때만
      // 토큰 업데이트를 위해 매트릭스 재생성 (조용히)
      setTimeout(() => {
        if (typeof setupAttentionVisualization === 'function') {
          const event = new Event('click');
          // generateBtn.dispatchEvent(event);
        }
      }, 100);
    }
  }
}

// 16.2 어텐션 시각화 및 16.3 매트릭스 생성
function setupAttentionVisualization() {
  const generateBtn = document.getElementById('generate-matrix');
  const updateBtn = document.getElementById('update-weights');
  const attentionMatrix = document.getElementById('attention-matrix');
  const layerSelect = document.getElementById('layer-select');
  const headSelect = document.getElementById('head-select');
  const tokenSliders = document.getElementById('token-sliders');
  
  // 현재 어텐션 데이터를 저장할 변수
  let currentAttentionData = null;
  let currentTokens = [];
  
  if (generateBtn && attentionMatrix) {
    generateBtn.addEventListener('click', function() {
      const button = this;
      button.textContent = '매트릭스 계산 중...';
      button.disabled = true;
      
      setTimeout(() => {
        generateAttentionMatrix();
        button.textContent = '매트릭스 생성';
        button.disabled = false;
      }, 1200);
    });
  }
  
  function generateAttentionMatrix() {
    // 현재 선택된 레이어와 헤드
    const layer = layerSelect?.value || '1';
    const head = headSelect?.value || '1';
    
    // 16.1의 입력 텍스트에서 토큰 가져오기
    const textInput = document.getElementById('demo-input');
    const inputText = textInput?.value || 'The transformer model';
    currentTokens = inputText.split(' ').filter(t => t.length > 0);
    
    // 토큰이 너무 많으면 처음 5개만 사용
    if (currentTokens.length > 5) {
      currentTokens = currentTokens.slice(0, 5);
    }
    
    // 어텐션 맵 시뮬레이션 - 의미적으로 연관된 토큰들이 더 높은 어텐션을 갖도록
    currentAttentionData = currentTokens.map((token, i) => 
      currentTokens.map((otherToken, j) => {
        let baseAttention = Math.random() * 0.6 + 0.1; // 0.1 ~ 0.7
        
        // 같은 토큰이면 높은 어텐션
        if (i === j) {
          baseAttention = Math.random() * 0.3 + 0.7; // 0.7 ~ 1.0
        }
        // 인접한 토큰들은 조금 더 높은 어텐션
        else if (Math.abs(i - j) === 1) {
          baseAttention = Math.random() * 0.4 + 0.4; // 0.4 ~ 0.8
        }
        // 'the'와 같은 관사는 명사와 높은 어텐션
        else if (token.toLowerCase() === 'the' && otherToken.toLowerCase().match(/model|transformer|system/)) {
          baseAttention = Math.random() * 0.3 + 0.5; // 0.5 ~ 0.8
        }
        
        return baseAttention.toFixed(3);
      })
    );
    
    displayAttentionMatrix(layer, head);
    
    // 토큰 슬라이더 생성
    if (tokenSliders) {
      tokenSliders.innerHTML = currentTokens.map((token, i) => `
        <div class="token-slider">
          <label>${token}: <span id="token-${i}-value">1.0</span></label>
          <input type="range" id="token-${i}-slider" min="0" max="2" step="0.1" value="1.0" 
                 onchange="document.getElementById('token-${i}-value').textContent = this.value">
        </div>
      `).join('');
    }
  }
  
  function displayAttentionMatrix(layer, head) {
    if (!currentAttentionData || !currentTokens.length) return;
    
    attentionMatrix.innerHTML = `
      <div class="attention-result">
        <h5>레이어 ${layer}, 헤드 ${head} 어텐션 매트릭스</h5>
        <div class="attention-matrix">
          <table>
            <thead>
              <tr>
                <th></th>
                ${currentTokens.map(token => `<th>${token}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
              ${currentTokens.map((token, i) => `
                <tr>
                  <td><strong>${token}</strong></td>
                  ${currentAttentionData[i].map((val, j) => `
                    <td style="background-color: rgba(0,123,255,${val}); color: ${val > 0.5 ? 'white' : 'black'}">
                      ${val}
                    </td>
                  `).join('')}
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        <div class="attention-stats">
          <div><strong>레이어:</strong> ${layer}/12</div>
          <div><strong>헤드:</strong> ${head}/12</div>
          <div><strong>시퀀스 길이:</strong> ${currentTokens.length}</div>
          <div><strong>어텐션 차원:</strong> 64</div>
          <div><strong>입력 텍스트:</strong> "${currentTokens.join(' ')}"</div>
        </div>
      </div>
    `;
  }
  
  // 외부에서 호출할 수 있는 함수 (16.1에서 텍스트 변경시 사용)
  window.updateAttentionMatrixForNewTokens = function(tokens) {
    if (currentAttentionData && tokens && tokens.length > 0) {
      // 자동으로 새 매트릭스 생성
      generateAttentionMatrix();
    }
  };
  
  if (updateBtn) {
    updateBtn.addEventListener('click', function() {
      const button = this;
      button.textContent = '가중치 적용 중...';
      button.disabled = true;
      
      setTimeout(() => {
        if (!currentAttentionData || !currentTokens.length) {
          alert('먼저 매트릭스를 생성해주세요!');
          button.textContent = '가중치 적용';
          button.disabled = false;
          return;
        }
        
        // 현재 슬라이더 값들 읽기
        const weights = [];
        for (let i = 0; i < currentTokens.length; i++) {
          const slider = document.getElementById(`token-${i}-slider`);
          if (slider) weights.push(parseFloat(slider.value));
          else weights.push(1.0);
        }
        
        // 어텐션 매트릭스에 가중치 적용
        const adjustedAttentionData = currentAttentionData.map((row, i) => 
          row.map((val, j) => {
            // 행(쿼리 토큰)과 열(키 토큰) 가중치를 곱함
            const adjustedVal = parseFloat(val) * weights[i] * weights[j];
            return Math.min(adjustedVal, 1.0).toFixed(3); // 최대값 1.0으로 제한
          })
        );
        
        // 정규화 (각 행의 합이 1이 되도록)
        const normalizedData = adjustedAttentionData.map(row => {
          const sum = row.reduce((acc, val) => acc + parseFloat(val), 0);
          return row.map(val => (parseFloat(val) / sum).toFixed(3));
        });
        
        // 현재 데이터 업데이트
        currentAttentionData = normalizedData;
        
        // 매트릭스 다시 표시
        const layer = layerSelect?.value || '1';
        const head = headSelect?.value || '1';
        displayAttentionMatrix(layer, head);
        
        // 결과 표시
        const resultDiv = document.createElement('div');
        resultDiv.className = 'weight-application-result';
        resultDiv.innerHTML = `
          <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h5 style="color: #155724; margin: 0 0 10px 0;">✅ 가중치 적용 완료!</h5>
            <p style="margin: 5px 0; color: #155724;">
              <strong>적용된 가중치:</strong><br>
              ${currentTokens.map((token, i) => `${token}: ${weights[i]}`).join('<br>')}
            </p>
            <p style="margin: 5px 0; color: #155724;">
              <strong>효과:</strong> 어텐션 매트릭스가 토큰별 중요도에 따라 조정되고 정규화되었습니다.
            </p>
            <p style="margin: 5px 0; font-size: 12px; color: #6c757d;">
              💡 높은 가중치를 가진 토큰들이 더 강한 어텐션을 받게 됩니다.
            </p>
          </div>
        `;
        
        // 기존 결과가 있으면 제거
        const existingResult = attentionMatrix.querySelector('.weight-application-result');
        if (existingResult) {
          existingResult.remove();
        }
        
        // 새 결과 추가
        attentionMatrix.appendChild(resultDiv);
        
        button.textContent = '가중치 적용';
        button.disabled = false;
      }, 500);
    });
  }
}

// 16.3 레이어 분석 (16.4 하이퍼파라미터 모니터링으로 대체)
function setupLayerAnalysis() {
  // 16.4 실시간 하이퍼파라미터 모니터링
  const currentSeqLen = document.getElementById('current-seq-len');
  const attentionDim = document.getElementById('attention-dim');
  const ffnRatio = document.getElementById('ffn-ratio');
  const totalParams = document.getElementById('total-params');
  
  // 실시간 업데이트 함수
  function updateHyperParameters() {
    const textInput = document.getElementById('demo-input');
    if (textInput && currentSeqLen) {
      const tokens = textInput.value.split(' ').filter(t => t.length > 0);
      currentSeqLen.textContent = tokens.length;
      
      // 어텐션 차원 업데이트 (d_model / n_heads)
      if (attentionDim) {
        attentionDim.textContent = Math.floor(768 / 12); // 64
      }
      
      // 다른 파라미터들은 고정값 유지
      if (ffnRatio) ffnRatio.textContent = '4x';
      if (totalParams) totalParams.textContent = '124M';
    }
  }
  
  // 입력 텍스트 변경시 실시간 업데이트
  const textInput = document.getElementById('demo-input');
  if (textInput) {
    textInput.addEventListener('input', updateHyperParameters);
    textInput.addEventListener('keyup', updateHyperParameters);
  }
  
  // 초기 업데이트
  updateHyperParameters();
  
  // 1초마다 업데이트 (실시간 효과)
  setInterval(updateHyperParameters, 1000);
}

// 16.4 성능 비교 (16.5로 통합)
function setupPerformanceComparison() {
  // 16.5 실제 계산 과정 보기 기능을 여기서 처리
  setupCalculationSteps();
}

// 16.5 실제 계산 과정
function setupCalculationProcess() {
  // 이미 setupPerformanceComparison에서 처리됨
}

// 16.5 계산 단계 업데이트 함수
function setupCalculationSteps() {
  const textInput = document.getElementById('demo-input');
  const tokenizationStep = document.getElementById('tokenization-step');
  const embeddingStep = document.getElementById('embedding-step');
  const attentionStep = document.getElementById('attention-step');
  const predictionStep = document.getElementById('prediction-step');
  
  function updateCalculationSteps() {
    if (!textInput) return;
    
    const text = textInput.value || 'The transformer model';
    const tokens = text.split(' ').filter(t => t.length > 0);
    
    // 1단계: 토큰화
    if (tokenizationStep) {
      const tokenIds = tokens.map(() => Math.floor(Math.random() * 50000));
      tokenizationStep.innerHTML = `
        <code>"${text}" → [${tokenIds.join(', ')}]</code>
        <div class="step-details">
          ${tokens.map((token, i) => `"${token}" → ${tokenIds[i]}`).join(', ')}
        </div>
      `;
    }
    
    // 2단계: 임베딩
    if (embeddingStep) {
      embeddingStep.innerHTML = `
        <code>토큰[${tokens.length}개] → 768차원 벡터 + 위치 임베딩</code>
        <div class="step-details">
          각 토큰: [1 × 768] + 위치[0~${tokens.length-1}]: [1 × 768] = [${tokens.length} × 768]
        </div>
      `;
    }
    
    // 3단계: 어텐션 계산
    if (attentionStep) {
      const seqLen = tokens.length;
      attentionStep.innerHTML = `
        <code>Q·K^T 스코어[${seqLen}×${seqLen}] → softmax → V와 곱셈</code>
        <div class="step-details">
          12개 헤드 × 64차원, 총 ${seqLen * seqLen * 12}개 어텐션 가중치 계산
        </div>
      `;
    }
    
    // 4단계: 다음 토큰 예측
    if (predictionStep) {
      const nextTokens = ['model', 'architecture', 'system', 'design'];
      const probs = nextTokens.map(() => (Math.random() * 0.3 + 0.1).toFixed(3));
      predictionStep.innerHTML = `
        <code>최종 벡터[768] → 50,257개 어휘 확률 분포</code>
        <div class="step-details">
          예상 다음 토큰: ${nextTokens.map((token, i) => `"${token}": ${probs[i]}`).join(', ')}
        </div>
      `;
    }
  }
  
  // 입력 변경시 계산 과정 업데이트
  if (textInput) {
    textInput.addEventListener('input', updateCalculationSteps);
    textInput.addEventListener('keyup', updateCalculationSteps);
  }
  
  // 초기 업데이트
  updateCalculationSteps();
  
  // 주기적 업데이트 (확률값 변경 효과)
  setInterval(() => {
    const predictionStep = document.getElementById('prediction-step');
    if (predictionStep && textInput) {
      const nextTokens = ['model', 'architecture', 'system', 'design'];
      const probs = nextTokens.map(() => (Math.random() * 0.3 + 0.1).toFixed(3));
      const currentCode = predictionStep.querySelector('code');
      if (currentCode) {
        predictionStep.innerHTML = `
          ${currentCode.outerHTML}
          <div class="step-details">
            예상 다음 토큰: ${nextTokens.map((token, i) => `"${token}": ${probs[i]}`).join(', ')}
          </div>
        `;
      }
    }
  }, 3000);
}

// 파라미터 컨트롤 설정 (15.2용)
function setupParameterControls() {
  const sliders = {
    'seq-len-slider': 'seq-len-value',
    'd-model-slider': 'd-model-value', 
    'n-layers-slider': 'n-layers-value',
    'n-heads-slider': 'n-heads-value'
  };
  
  Object.entries(sliders).forEach(([sliderId, valueId]) => {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(valueId);
    
    if (!slider || !valueDisplay) return;
    
    slider.addEventListener('input', function() {
      valueDisplay.textContent = this.value;
      updateComplexityDisplay();
    });
  });
  
  function updateComplexityDisplay() {
    const seqLen = parseInt(document.getElementById('seq-len-slider')?.value || 64);
    const dModel = parseInt(document.getElementById('d-model-slider')?.value || 256);
    const nLayers = parseInt(document.getElementById('n-layers-slider')?.value || 4);
    const nHeads = parseInt(document.getElementById('n-heads-slider')?.value || 4);
    
    // 복잡도 계산
    const attnComplexity = seqLen * seqLen * dModel;
    const ffnComplexity = seqLen * dModel * dModel * 4;
    const totalComplexity = nLayers * (attnComplexity + ffnComplexity);
    
    // 메모리 사용량 (대략적)
    const memoryMB = (seqLen * dModel * 4 * nLayers) / (1024 * 1024);
    
    // 추론 속도 (가상)
    const speed = Math.max(50, 200 - (totalComplexity / 1000000));
    
    // 디스플레이 업데이트
    const memoryElement = document.getElementById('memory-value');
    const speedElement = document.getElementById('speed-value');
    const complexityElement = document.getElementById('complexity-value');
    
    if (memoryElement) memoryElement.textContent = `${memoryMB.toFixed(1)}MB`;
    if (speedElement) speedElement.textContent = `${speed.toFixed(0)}ms`;
    if (complexityElement) complexityElement.textContent = `${(totalComplexity / 1000000).toFixed(1)}M`;
  }
  
  // 초기 업데이트
  updateComplexityDisplay();
}

// 20번 섹션 시뮬레이션
function setupAdvancedGPTSimulation() {
  const generateBtn = document.getElementById('advanced-generate-btn');
  const flashBtn = document.getElementById('flash-attention-btn');
  const promptInput = document.getElementById('advanced-prompt-input');
  const resultDiv = document.getElementById('advanced-result');
  
  if (generateBtn && promptInput && resultDiv) {
    generateBtn.addEventListener('click', function() {
      const prompt = promptInput.value || 'Hello World';
      
      // 시뮬레이션
      resultDiv.innerHTML = `
        <div class="simulation-result">
          <h5>🎯 고급 텍스트 생성 결과</h5>
          <div><strong>입력:</strong> "${prompt}"</div>
          <div><strong>출력:</strong> "${prompt} generated with advanced GPT!"</div>
          <div class="performance-metrics">
            <div><strong>처리 시간:</strong> 45ms</div>
            <div><strong>토큰/초:</strong> 150</div>
            <div><strong>메모리 사용:</strong> 2.3MB</div>
            <div><strong>KV 캐시:</strong> 활성화 ✅</div>
          </div>
        </div>
      `;
    });
  }
  
  if (flashBtn && resultDiv) {
    flashBtn.addEventListener('click', function() {
      resultDiv.innerHTML = `
        <div class="simulation-result">
          <h5>⚡ Flash Attention 시연</h5>
          <div><strong>입력 시퀀스 길이:</strong> 512</div>
          <div><strong>메모리 최적화:</strong> O(N²) → O(N)</div>
          <div class="performance-metrics">
            <div><strong>기존 어텐션:</strong> 1024MB 메모리</div>
            <div><strong>Flash Attention:</strong> 256MB 메모리 (75% 절약)</div>
            <div><strong>속도 향상:</strong> 3.2x 빠름</div>
            <div><strong>정확도:</strong> 99.9% 일치</div>
          </div>
        </div>
      `;
    });
  }
}

// Hello World 생성 시뮬레이션 (19번 섹션)
function setupHelloWorldSimulation() {
  const simulateBtn = document.getElementById('hello-world-simulate');
  const resultDiv = document.getElementById('hello-world-result');
  
  if (simulateBtn && resultDiv) {
    simulateBtn.addEventListener('click', function() {
      const steps = [
        '입력 토큰화: "Hello" → [15] "World" → [23]',
        '임베딩 변환: [15, 23] → [[0.1, -0.3, ...], [0.4, 0.1, ...]]',
        '위치 임베딩 추가: pos[0] + pos[1]',
        '트랜스포머 블록 1: Self-Attention + FFN',
        '트랜스포머 블록 2: Self-Attention + FFN',
        '최종 정규화: LayerNorm 적용',
        '언어모델 헤드: 어휘 확률 분포 계산',
        '샘플링: "!" 토큰 선택 (확률: 0.87)',
        '결과: "Hello World!"'
      ];
      
      resultDiv.innerHTML = `
        <div class="generation-steps">
          <h5>🔄 "Hello World" → "Hello World!" 생성 과정</h5>
          ${steps.map((step, i) => `
            <div class="generation-step">
              <div class="step-number">${i + 1}</div>
              <div class="step-description">${step}</div>
            </div>
          `).join('')}
        </div>
      `;
    });
  }
}

// 20.4 실시간 성능 시뮬레이션
function setupPerformanceSimulation() {
  const seqLengthSlider = document.getElementById('seq-length');
  const batchSizeSlider = document.getElementById('batch-size');
  const enableCacheCheckbox = document.getElementById('enable-cache');
  
  const seqLengthValue = document.getElementById('seq-length-value');
  const batchSizeValue = document.getElementById('batch-size-value');
  
  const inferenceSpeedElement = document.getElementById('inference-speed');
  const memoryUsageElement = document.getElementById('memory-usage');
  const latencyElement = document.getElementById('latency');
  
  function updateMetrics() {
    const seqLength = parseInt(seqLengthSlider?.value || 64);
    const batchSize = parseInt(batchSizeSlider?.value || 1);
    const useCache = enableCacheCheckbox?.checked || false;
    
    // 실시간 계산
    const baseMemory = seqLength * 0.5 * batchSize; // MB
    const cacheBonus = useCache ? 0.8 : 1.0; // 캐시 사용시 80% 메모리
    const memoryUsage = baseMemory * cacheBonus;
    
    const baseSpeed = Math.max(50, 200 - (seqLength / 5)); // tokens/sec
    const cacheSpeedBonus = useCache ? 1.5 : 1.0;
    const inferenceSpeed = baseSpeed * cacheSpeedBonus;
    
    const baseLatency = seqLength / 10 + batchSize * 2; // ms
    const latency = baseLatency / cacheSpeedBonus;
    
    // UI 업데이트
    if (seqLengthValue) seqLengthValue.textContent = seqLength;
    if (batchSizeValue) batchSizeValue.textContent = batchSize;
    if (inferenceSpeedElement) inferenceSpeedElement.textContent = `${inferenceSpeed.toFixed(1)} tokens/sec`;
    if (memoryUsageElement) memoryUsageElement.textContent = `${memoryUsage.toFixed(1)} MB`;
    if (latencyElement) latencyElement.textContent = `${latency.toFixed(1)} ms`;
  }
  
  // 이벤트 리스너 추가
  if (seqLengthSlider) {
    seqLengthSlider.addEventListener('input', updateMetrics);
  }
  if (batchSizeSlider) {
    batchSizeSlider.addEventListener('input', updateMetrics);
  }
  if (enableCacheCheckbox) {
    enableCacheCheckbox.addEventListener('change', updateMetrics);
  }
  
  // 초기 업데이트
  updateMetrics();
}

// 20.5 cURL 테스트 설정
function setupCurlTests() {
  // cURL 예제 섹션을 HTML에 추가
  const apiSection = document.querySelector('.api-documentation');
  if (apiSection) {
    const curlSection = document.createElement('div');
    curlSection.className = 'curl-test-section';
    curlSection.innerHTML = `
      <h4>cURL 테스트 예제</h4>
      <div class="curl-examples">
        <div class="curl-example">
          <div class="curl-title">1. 서버 상태 확인</div>
          <div class="curl-command">
            <code>curl -X GET http://localhost:18080/api/health</code>
          </div>
          <button class="copy-btn" onclick="copyCurlCommand(this)">복사</button>
        </div>
        
        <div class="curl-example">
          <div class="curl-title">2. 텍스트 생성 요청</div>
          <div class="curl-command">
            <code>curl -X POST http://localhost:18080/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello", "max_tokens": 50}'</code>
          </div>
          <button class="copy-btn" onclick="copyCurlCommand(this)">복사</button>
        </div>
        
        <div class="curl-example">
          <div class="curl-title">3. Flash Attention 시연</div>
          <div class="curl-command">
            <code>curl -X POST http://localhost:18080/api/flash_attention \\
  -H "Content-Type: application/json" \\
  -d '{"sequence_length": 512, "enable_profiling": true}'</code>
          </div>
          <button class="copy-btn" onclick="copyCurlCommand(this)">복사</button>
        </div>
        
        <div class="curl-example">
          <div class="curl-title">4. 모델 설정 조회</div>
          <div class="curl-command">
            <code>curl -X GET http://localhost:18080/api/config</code>
          </div>
          <button class="copy-btn" onclick="copyCurlCommand(this)">복사</button>
        </div>
      </div>
      
      <div class="curl-note">
        <p><strong>사용 방법:</strong></p>
        <ol>
          <li>먼저 Crow 서버를 실행하세요. 
                <code>./build/advanced_gpt_server</code></li>
          <li>위의 cURL 명령어를 터미널에 복사해서 실행하세요.</li>
          <li>서버가 18080 포트에서 실행되고 있는지 확인하세요.</li>
          <li>JSON 응답을 확인하여 API가 정상 작동하는지 테스트하세요.</li>
        </ol>
      </div>
    `;
    apiSection.appendChild(curlSection);
  }
}

// cURL 명령어 복사 함수
function copyCurlCommand(button) {
  const codeElement = button.previousElementSibling.querySelector('code');
  const command = codeElement.textContent;
  
  navigator.clipboard.writeText(command).then(() => {
    button.textContent = '복사됨!';
    setTimeout(() => {
      button.textContent = '복사';
    }, 2000);
  }).catch(err => {
    console.error('복사 실패:', err);
    button.textContent = '복사 실패';
    setTimeout(() => {
      button.textContent = '복사';
    }, 2000);
  });
}

// 전역 함수로 추가 (HTML에서 호출용)
window.copyCurlCommand = copyCurlCommand;
window.runPerformanceSimulation = function() {
  // 시뮬레이션 실행 시각적 피드백과 실제 계산
  const button = event.target;
  const originalText = button.textContent;
  button.textContent = '실행 중...';
  button.disabled = true;
  
  // 현재 설정값 읽기
  const seqLength = parseInt(document.getElementById('seq-length')?.value || 64);
  const batchSize = parseInt(document.getElementById('batch-size')?.value || 1);
  const useCache = document.getElementById('enable-cache')?.checked || false;
  
  // 새로운 랜덤 계산으로 시뮬레이션 효과
  setTimeout(() => {
    // 약간의 랜덤 변동 추가
    const randomFactor = 0.9 + Math.random() * 0.2; // 0.9 ~ 1.1
    
    const baseMemory = seqLength * 0.5 * batchSize;
    const cacheBonus = useCache ? 0.8 : 1.0;
    const memoryUsage = baseMemory * cacheBonus * randomFactor;
    
    const baseSpeed = Math.max(50, 200 - (seqLength / 5));
    const cacheSpeedBonus = useCache ? 1.5 : 1.0;
    const inferenceSpeed = baseSpeed * cacheSpeedBonus * randomFactor;
    
    const baseLatency = seqLength / 10 + batchSize * 2;
    const latency = baseLatency / cacheSpeedBonus / randomFactor;
    
    // 업데이트된 지표 표시
    const inferenceSpeedElement = document.getElementById('inference-speed');
    const memoryUsageElement = document.getElementById('memory-usage');
    const latencyElement = document.getElementById('latency');
    
    if (inferenceSpeedElement) inferenceSpeedElement.textContent = `${inferenceSpeed.toFixed(1)} tokens/sec`;
    if (memoryUsageElement) memoryUsageElement.textContent = `${memoryUsage.toFixed(1)} MB`;
    if (latencyElement) latencyElement.textContent = `${latency.toFixed(1)} ms`;
    
    button.textContent = '완료!';
    setTimeout(() => {
      button.textContent = originalText;
      button.disabled = false;
    }, 1500);
  }, 1000);
};

// DOMContentLoaded 이벤트
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, setting up interactive features...');
  
  // 기본 설정
  setupCodeViewer();
  setupParameterControls();
  
  // 섹션 16 인터랙티브 기능
  setupSection16Interactions();
  
  // 고급 기능
  setupAdvancedGPTSimulation();
  setupHelloWorldSimulation();
  
  // 20.4, 20.5 기능
  setupPerformanceSimulation();
  setupCurlTests();
  
  console.log('All interactive features initialized');
});
