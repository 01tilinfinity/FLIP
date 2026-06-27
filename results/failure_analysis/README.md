# ExcluIR 실패 사례 분석 (target_minus_trap, rewriter=gpt-4o-mini)

원본 데이터: `results/score_matrices/excluir_1000_*_rewriter_gpt4o_mini*.npz`,
샘플 `data/excluir_manual_1000_seed42.csv`,
디컴포지션 `outputs/excluir_rewriter_gpt4o_mini/decompositions.jsonl`.
`results/excluir_embedding_model_comparison_rewriter_gpt4o_mini/`에서 가져온
모델별 최적 `target_minus_trap` 설정(candidate_top_n=all):

| 모델 | gamma | beta |
|---|---|---|
| qwen3-embedding-4b | 1.0 | 0.5 |
| qwen3-embedding-0.6b | 0.25 | 0.1 |
| bge-m3 | 0.25 | 0.1 |
| text-embedding-3-small | 1.0 | 0.5 |

k=5 기준 실패 정의: **recall 손실(recall dropped)** = baseline에서 정답 문서
rank ≤5였지만, (target − trap) 적용 후 rank가 >5로 떨어진 경우.
**trap 잔존(trap persists)** = penalty 적용 후에도 trap 문서가 여전히 rank ≤5인
경우. 전체 예시 표(모델당 최대 50개)는 `recall_dropped_examples.csv`와
`trap_persists_examples.csv`에 있습니다.

1000개 쿼리 중 실패 건수:

| 모델 | recall 손실 | trap 잔존 |
|---|---|---|
| qwen3-embedding-4b | 59 | 87 |
| qwen3-embedding-0.6b | 53 | 98 |
| bge-m3 | 89 | 102 |
| text-embedding-3-small | 97 | 89 |

## 발견된 실패 모드

**1. 정답 문서 안에 trap 내용이 섞여 있음 (recall 손실의 주된 원인).**
ExcluIR의 제외 조건은 대부분 "X인데 Y는 빼고"처럼, X와 Y가 같은 글/같은
엔티티 군에 강하게 묶여 있는 형태로 표현됩니다. 그래서 정답 문서 본문에
*제외 대상 엔티티가 부수적으로 언급*되어 있는 경우가 많습니다. 예시
`excluir_31`: 정답 문서는 aTelecine에 관한 글이지만, 본문에 "공동 창립자는
... Sasha Grey"라는 문장이 들어 있습니다 — 이게 바로 제외하려던 엔티티입니다.
`sim(q_trap, doc)`로 패널티를 주면 정답 문서 자체가 같이 깎입니다. qwen3-4b
기준 rank가 1 → 25로 떨어지고, 극단적인 경우(`excluir_736`, Marlborough
House / Queen Alexandra Memorial)는 1 → 57,577까지 붕괴합니다. 이 패턴
하나만으로 4개 모델 전체에서 가장 심각한 recall 손실 사례들이 거의 다
설명됩니다.

**2. 정답 문서와 trap 문서가 임베딩 공간에서 거의 동일한 위치에 있음.**
많은 trap 문서가 같은 주제의 형제/대조 엔티티에 관한 글입니다 (예:
`excluir_0`: "Annie (1982)" vs "Annie (브로드웨이 각색판)"; `excluir_2638`:
Sonny Moorman의 스타일 vs "Eric Clapton 밴드에 비유됨"). `sim(q_target, ·)`
기준으로 보면 이 둘의 임베딩이 거의 겹쳐 있어서, q_trap을 잘 만들어도
정답 문서 점수가 trap 문서 점수만큼 같이 떨어집니다. 그래서 "trap 잔존"
사례들은 대부분 penalty가 안 먹힌 게 아니라, trap 문서가 정답 문서와
나란히 rank 1-5 안에 같이 남아 있는 형태로 나타납니다.

**3. Q_trap이 원래 제외 조건보다 더 넓게 잡히는 경우.**
일부 Q_trap이 제외 대상 자체보다 그 도메인 전체에 걸쳐 있는 핵심 문구를
다시 가져오는 경우가 있습니다 (예: `excluir_1421`: Q_target = "Los Angeles
Rams에서 뛴 유명 미국 풋볼 선수", Q_trap = "Eric Dickerson" — Dickerson 본인도
Rams 선수이기 때문에, trap 쿼리가 정답에 가까운 다른 Rams 관련 문서들까지
같이 걸러버립니다; rank 5 → 454). 이는 핸드오프에서 우려했던 "q_trap이
positive 개념과 겹친다"는 위험과 정확히 일치합니다.

**4. Q_target이 가끔 지나치게 일반적으로 변환되어 엔티티 특정성을 잃음.**
이 샘플에서는 빈도가 낮지만, 일부 Q_target이 정답 문서를 다른 비슷한
코퍼스 문서들과 구분해주는 고유명사/수식어를 빠뜨리는 경우가 있습니다
(baseline rank가 이미 애매했던 사례들에서 주로 관찰됨, 예: `excluir_1484`,
rank 5 → 19).

1번과 2번이 압도적으로 많습니다 — 두 표 모두에서 대부분의 사례를 차지하며,
이는 *데이터셋 구조* 자체의 특성(정답/trap 문서가 설계상 같은 엔티티나
주제를 공유함)이고, rewriter 프롬프트 문제가 아닙니다. 고정 가중치 선형
penalty(`gamma*target - beta*trap`)는 어떤 임베딩 모델을 쓰든 거의 같은
recall-violation 트레이드오프 곡선을 그릴 수밖에 없는데, 실제로
embedding-model-comparison 요약에서도 4개 모델 모두 각자의 최적 설정에서
동일한 recall 손실 패턴을 보입니다.

## 프롬프트 변형에 대한 시사점

핸드오프의 Variant 1-4(recall 보존형 target, trap 특정성 강화, anti-overlap
체크, 보수적 rewrite)는 3번/4번 모드(rewriter 표현 문제)는 개선할 수 있지만,
**1번/2번 모드는 고칠 수 없습니다**. 이건 쿼리 재작성이 아니라 문서 내용
자체에서 비롯된 문제이기 때문입니다. 1번/2번을 해결하려면 다음 중 하나가
필요합니다:
- 비선형 penalty(threshold/hinge 방식, 섹션 12D 참고) — trap 유사도가
  target 유사도를 명확히 넘어서기 전까지는 패널티를 주지 않는 방식, 또는
- 문서 단위 체크 — 문서 전체의 풀링된 임베딩 하나가 아니라, 후보 문서에서
  실제로 trap 전용 내용에 해당하는 부분이 있는지 확인하는 방식.

이 결과를 바탕으로 한 제안: 우리가 구체적인 사례(`excluir_1421` 류)를 확인한
3번 모드(rewriter로 고칠 수 있는 유일한 모드)를 직접 타겟하는 Variant 3
(anti-overlap)를 먼저 시도하고, threshold/hinge penalty(섹션 12D)는 1번/2번
모드에 대해 더 효과가 클 가능성이 높은 별도의 후속 작업으로 진행하는 것을
권장합니다.
