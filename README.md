# GPT(Decoder-only, Pre‑LN) 최소 구현 — 표준 아키텍처

> 본 README는 `web/index.html`의 본문과 동일한 내용을 텍스트로 제공합니다.

## 개요
- GPT 계열(Decoder-only, Pre-LN) 표준 구조를 따릅니다.
- 임베딩(학습형 위치), 멀티헤드 *인과적* 자기어텐션, FFN(GELU), 최종 LayerNorm, Proj(d→V).
- 드롭아웃: embedding / attention / residual만. FFN 내부 드롭아웃은 사용하지 않음.

## 표기와 차원
- 배치 \(B\), 길이 \(L\), 모델차원 \(d\), 헤드수 \(h\), 헤드차원 \(d_h=d/h\), 어휘 \(V\).
- 임베딩: \(X^{(0)} = \text{Lookup}(T,E) + P\),  \(X^{(0)}\in\mathbb{R}^{B\times L\times d}\).

## 수식
- Scaled Dot-Product (causal): 
  - \(S=QK^\top/\sqrt{d_h}\), \(\tilde S=S+M\) (상삼각 \(-\infty\)), \(A=\operatorname{softmax}(\tilde S)\), \(O=AV\).
- Pre-LN 블록: 
  - \(\hat X=\text{LN}(X);\, X'=X+\text{Drop}(\text{Attn}(\hat X));\, \hat X'=\text{LN}(X');\, X^+=X'+\text{Drop}(\text{FFN}(\hat X'))\).
- FFN(GELU): \(H=\text{GELU}(X W_1+b_1);\, Y=H W_2 + b_2\).

## 구현 요약
- 학습형 위치임베딩, 임베딩 스케일링 없음.
- Causal mask 필수, 마지막 LayerNorm 후 Proj(d→V).
- FFN: d→4d→d, 활성화 GELU, FFN dropout 없음.

## Back(Crow) 개요
- `GET /api/health`: 상태 확인
- `GET /api/config`: 하이퍼파라미터
- `POST /api/attention`: 표준 수식으로 단일 배치/헤드 attention 계산 데모
# mini-gpt
