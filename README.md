# mini-gpt: C++ 기반 GPT(Decoder-only, Pre-LN) 최소 구현

## 프로젝트 개요

- **목표:** 논문 수준의 GPT(Decoder-only, Pre-LN) 아키텍처를 C++로 완전하게 구현하고, 수식/코드/이론을 1:1로 매핑하여 누구나 재현·확장할 수 있도록 공개합니다.
- **특징:**  
  - 학습형 위치 임베딩, 멀티헤드 인과적(Self-Attention) 구조, FFN(GELU), Pre-LayerNorm, 최종 Proj(d→V)
  - 드롭아웃: embedding / attention / residual만 적용 (FFN 내부 드롭아웃 없음)
  - Crow C++ 웹 프레임워크 기반 RESTful API 제공
  - 수식, 차원, 메모리 복잡도, 성능 최적화까지 논문 수준으로 명확히 기술

## 주요 수식 및 차원

- **표기:**  
  - 배치 \(B\), 길이 \(L\), 모델차원 \(d\), 헤드수 \(h\), 헤드차원 \(d_h=d/h\), 어휘 \(V\)
  - 임베딩: \(X^{(0)} = \text{Lookup}(T,E) + P\),  \(X^{(0)}\in\mathbb{R}^{B\times L\times d}\)
- **Scaled Dot-Product Attention:**  
  - \(S=QK^\top/\sqrt{d_h}\), \(\tilde S=S+M\) (상삼각 \(-\infty\)), \(A=\operatorname{softmax}(\tilde S)\), \(O=AV\)
- **Pre-LN 블록:**  
  - \(\hat X=\text{LN}(X);\, X'=X+\text{Drop}(\text{Attn}(\hat X));\, \hat X'=\text{LN}(X');\, X^+=X'+\text{Drop}(\text{FFN}(\hat X'))\)
- **FFN(GELU):**  
  - \(H=\text{GELU}(X W_1+b_1);\, Y=H W_2 + b_2\)

## 폴더 구조
```
$ tree -I third_party
.
├── back                        # C++ 백엔드 서버 및 GPT 모델 구현 폴더
│   ├── advanced_gpt.hpp        # GPT 모델 핵심 헤더 파일
│   ├── advanced_tensor.cpp     # 텐서 연산 구현 소스 파일
│   ├── CMakeLists.txt          # CMake 빌드 설정 파일
│   ├── include                 # 헤더 파일 모음 폴더
│   │   └── model               # 모델 관련 헤더 파일 폴더
│   │       ├── advanced_transformer.hpp # 고급 Transformer 구현 헤더
│   │       ├── attention.hpp   # Self-Attention 구현 헤더
│   │       ├── ffn.hpp         # FFN(GELU) 구현 헤더
│   │       ├── layernorm.hpp   # LayerNorm 구현 헤더
│   │       ├── tensor.hpp      # 텐서 자료구조 헤더
│   │       └── transformer.hpp # 기본 Transformer 헤더
│   └── main.cpp                # Crow 기반 서버 진입점
├── deploy.sh                   # 배포 자동화 스크립트
├── diagram.js                  # 모델 구조 시각화 JS 파일
├── index.html                  # 프로젝트 메인 웹페이지
├── models                      # 학습된 모델 및 체크포인트 폴더
├── README.md                   # 프로젝트 설명 문서
├── scripts.js                  # 웹페이지용 JS 스크립트
├── styles.css                  # 웹페이지용 CSS 스타일
├── test_kakao_env.php          # 카카오 환경 테스트용 PHP 스크립트
└── transformer-viewer          # Transformer 시각화 웹앱 폴더
  ├── app.js                  # 시각화 앱 JS 파일
  ├── index.html              # 시각화 앱 웹페이지
  └── styles.css              # 시각화 앱 CSS 스타일

6 directories, 20 files
```

## 빌드 및 실행

### Crow C++ 서버

```bash
cd back
mkdir -p build && cd build
cmake ..
make
./back
```

- 서버가 실행되면 `http://localhost:18080`에서 REST API 접근 가능

### OpenAI API 프록시(PHP)

- `.env.local`에 `OPENAI_API_KEY=sk-...` 추가
- PHP 서버에서 `api/chatgpt.php`가 동작해야 함

## REST API 예시

- `GET /api/health` : 서버 상태 확인
- `GET /api/config` : 하이퍼파라미터 정보
- `POST /api/attention` : 단일 배치/헤드 attention 계산 데모

## 주요 구현/최적화 포인트

- 학습형 위치임베딩, 임베딩 스케일링 없음
- Causal mask 필수, 마지막 LayerNorm 후 Proj(d→V)
- FFN: d→4d→d, 활성화 GELU, FFN dropout 없음
- SIMD, 메모리 정렬, 멀티스레딩 등 C++ 최적화 적용

## 참고 논문

- Vaswani et al., "Attention is All You Need" (2017)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- 기타 GPT-2, Pre-LN, FlashAttention 등 관련 논문

## 라이선스

- MIT License (오픈소스, 자유 사용/수정/배포 가능)

## 정리 홈페이지
https://minigpt.dothome.co.kr