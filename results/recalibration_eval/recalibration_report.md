# Exploration Recalibration Eval — Report

이번 수정은 short-term LLM이 recent 변화를 감지하는 능력 자체를 없애는 것이 아니라,
그 변화를 너무 쉽게 exploration으로 해석하던 문제를 교정하여,
대부분의 사례를 원래 취향(persona) 위에 최근 의도를 얹는
aligned_soft / task_focus_like / budget_like 상태로 재해석하는 작업이다.

---

## 1. Recalibrated Reason Distribution (exploration rows only)

| reason | before | after | delta |
|---|---|---|---|
| aligned_soft | 0 | 372 | +372 |
| budget_like | 0 | 279 | +279 |
| exploration | 1154 | 0 | -1154 |
| exploration_unclear | 0 | 187 | +187 |
| task_focus_like | 0 | 15 | +15 |
| true_exploration | 0 | 301 | +301 |

## 2. HR@10 — Branch Comparison

| branch | mode | HR@10 | NDCG@10 | MRR | gt_delta_pos | gt_delta_zero |
|---|---|---|---|---|---|---|
| A_current_baseline | full_model | 0.6015 | 0.4022 | 0.3566 | 0.9175 | 0.0815 |
| A_current_baseline | intent_only | 0.5875 | 0.3907 | 0.3463 | 0.083 | 0.917 |
| A_current_baseline | persona_only | 0.629 | 0.4236 | 0.3754 | 0.915 | 0.085 |
| B_recalibrated | full_model | 0.588 | 0.3904 | 0.3457 | 0.9175 | 0.0815 |
| B_recalibrated | intent_only | 0.5865 | 0.3898 | 0.3455 | 0.083 | 0.917 |
| B_recalibrated | persona_only | 0.629 | 0.4236 | 0.3754 | 0.915 | 0.085 |
| D_heuristic_control | full_model | 0.6145 | 0.4113 | 0.3638 | 0.9235 | 0.075 |
| D_heuristic_control | intent_only | 0.581 | 0.3822 | 0.3375 | 0.238 | 0.762 |
| D_heuristic_control | persona_only | 0.629 | 0.4236 | 0.3754 | 0.915 | 0.085 |

## 3. Exploration Redistribution

- Total exploration records: 1154
- Recalibrated away from exploration: 1154 (100.0%)

| recalibrated_reason | count | % of exploration |
|---|---|---|
| aligned_soft | 372 | 32.2% |
| true_exploration | 301 | 26.1% |
| budget_like | 279 | 24.2% |
| exploration_unclear | 187 | 16.2% |
| task_focus_like | 15 | 1.3% |

## 4. Semantic-Core Overlap Distribution

- mean sc_overlap (exploration rows): 0.135
- fraction with sc_overlap == 0.0: 0.560
- fraction with sc_overlap >= 0.4: 0.109

## 5. Conclusion

현재 PGIM의 문제는 exploration 신호 자체가 아니라,
exploration으로 과대 해석되는 reason calibration 문제였는가?

> **결론**: 위 비교 결과를 기반으로 판단 (B vs A HR@10 delta 참고)