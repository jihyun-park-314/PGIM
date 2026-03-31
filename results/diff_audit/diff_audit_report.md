# Raw vs Validated Diff Audit Report

## 핵심 진단

1. Stage 2 activation gate는 전체 38.0%의 레코드에서 goal set을 변경하며, exploration에서 56.2%로 가장 높다.

2. 그러나 변경된 케이스에서도 gt_match_delta=0.02267로 GT alignment는 거의 개선되지 않는다 — selector가 후보 공간 정합성은 높이지만 GT-discriminative selection은 수행하지 못한다.

3. Activation gate는 특이도가 높은(niche) concept를 제거하고 광범위한(generic) concept를 살리는 경향이 있다 (removed avg_spec=6.131 vs survived avg_spec=3.767).

## Aggregate Summary

| reason | n | changed% | avg_jaccard | gt_match_raw | gt_match_val | gt_match_delta | spec_raw | spec_val |
|---|---|---|---|---|---|---|---|---|
| aligned | 519 | 0.044 | 0.970 | 0.7280 | 0.7302 | 0.00225 | 1.245 | 1.106 |
| budget_shift | 74 | 0.392 | 0.773 | 0.1374 | 0.1622 | 0.02478 | 3.934 | 3.135 |
| exploration | 1154 | 0.562 | 0.634 | 0.0399 | 0.0487 | 0.00874 | 4.648 | 3.126 |
| task_focus | 243 | 0.226 | 0.826 | 0.1145 | 0.1255 | 0.01097 | 3.758 | 2.908 |
| unknown | 10 | 0.600 | 0.617 | 0.5500 | 0.7000 | 0.15001 | 1.114 | 0.667 |

## Hypothesis 판정

**H1 (selector mostly keeps same concepts):** CONFIRMED
- exploration unchanged rate: 0.438, avg jaccard: 0.634

**H2 (survived concepts are broader/generic):** CONFIRMED
- removed avg doc_freq=881.5 spec=6.131 vs survived avg doc_freq=6890.5 spec=3.767

**H3 (exploration backbone activation weak):** CONFIRMED
- exploration avg_act_val=3.0770 vs aligned=43.6574 (ratio=0.0705)

**H4 (gt_match barely improves even when changed):** CONFIRMED
- avg gt_match_delta (changed)=0.02267, delta=0 fraction=0.940

## Next Selector Redesign Direction

**Discriminative grounded selector** — activation gate에 specificity score를 추가.

현재 selector는 `activation_count >= 1`이면 통과시킨다. 결과적으로 high-frequency generic concept (doc_freq 높음, specificity 낮음)이 살아남고, 실제로 이 후보군을 특징짓는 specific concept는 activation이 낮아 제거된다.

제안: activation gate 통과 후 concept들을 `score = activation_count × specificity_weight`로 재정렬하여 generic concept에 암묵적 감점을 부여.
specificity_weight = log(N / doc_freq) — IDF와 동일한 형태로 기존 인프라 재활용 가능.

## Persona Ontology와 LLM Short-term Branch 상호작용

**Q1. 왜 intent-only에서는 LLM이 약간 낫고, full_model에서는 heuristic이 더 높은가?**

intent-only에서 LLM(+0.0065)이 앞서는 이유: LLM은 exploration/task_focus를 세밀하게 구분하여 modulation policy를 더 정확하게 적용한다. heuristic은 aligned를 48.7%로 과다 분류하여 많은 케이스에서 boost 기회를 놓친다.

full_model에서 heuristic(-0.013)이 앞서는 이유: LLM은 exploration을 57.7%로 과다 분류한다. exploration goal의 gt_delta zero_frac=0.923 — 거의 모든 경우 GT delta=0. 이는 persona graph의 안정적 prior 기여를 방해하지는 않지만, exploration으로 잘못 분류된 케이스에서 persona signal이 제대로 증폭되지 않아 signal dilution이 발생한다.

**Q2. LLM과 persona ontology가 본질적으로 안 맞는가, 아니면 조율 문제인가?**

조율 문제다. full_model에서 zero variant(exploration 완전 제거) 시 HR@10이 0.6015→0.5885로 급락한다. 즉 exploration signal 자체는 full_model에 기여하고 있다. 문제는 LLM이 exploration을 너무 많이 잡아 persona prior의 aligned/task_focus 기여가 exploration policy로 희석되는 것이다. LLM reason과 persona는 본질적으로 상보적 구조이나, 현재 exploration 과다 분류가 그 상보성을 노이즈로 바꾸고 있다.

**Q3. 상보적 긴장을 만들려면 selector/fusion에서 무엇이 필요한가?**

두 가지가 필요하다:
1. **Exploration-specific discriminative selection**: 현재처럼 activation만으로 통과시키면 generic concept만 남는다. specificity-weighted selection으로 persona prior와 실질적으로 다른 specific concept만 exploration signal에 포함시켜야 한다.
2. **Confidence-conditioned fusion weight**: exploration reason이어도 selected goal의 specificity/activation이 낮으면 persona prior를 더 신뢰하도록 gate strength를 조정. 현재 gate는 reason+confidence로 구동되지만, goal quality(specificity)를 추가 입력으로 받는 것이 필요하다.

## 핵심 질문에 대한 한 문장 답변

> 지금 PGIM의 핵심 문제는 exploration을 많이 잡는 것과 GT-discriminative signal로 변환하지 못하는 것이 **동시에** 존재하며, 전자(과다 분류)는 full_model에서 persona와의 조율 실패를 유발하고, 후자(변환 실패)는 intent-only의 ceiling을 낮추고 있다 — 그러나 두 문제 중 **selector의 GT-discriminative 변환 실패**가 더 근본적이다. exploration 비율을 줄여도 선택된 goal이 GT와 무관하면 ranking utility는 여전히 낮기 때문이다.
