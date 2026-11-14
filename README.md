# Does removing ground-truth feedback change what ACE learns?

A small controlled experiment inspired by **ACE** (Agentic Context Engineering), made for a CS329Z paper video. It is not a reproduction of the paper's numbers.

> Zhang et al. *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models.* ICLR 2026. [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)

## Question

Without the correct answer, can ACE's Reflector still write sound, reusable lessons? Do those lessons help or hurt on new questions?

## Setup

| Item | Choice |
| --- | --- |
| Task | [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) intent classification, restricted to 7 easily confused transfer intents |
| Data | 21 build questions (train split), 49 test questions (test split), balanced across intents, seed 329 |
| Model | `openai/gpt-4o-mini` via OpenRouter, temperature 0, shared by all roles |
| Implementation | Minimal ACE: Generator, Reflector, Curator, and code that appends new bullets. No helpful/harmful counters, de-duplication or multi-epoch |

| Condition | Playbook used at test time |
| --- | --- |
| A. Baseline | None |
| B. With labels | Built with the correct answer shown to the Reflector |
| C. Without labels | Built without the correct answer |

The Generator answers the 21 build questions once, with no playbook. B and C then reflect on exactly the same answers in the same order, so the only difference between them is whether the Reflector sees the correct answer.

## Results

| Condition | Correct / 49 | Bullets |
| --- | --- | --- |
| A. Baseline | 37 (75.5%) | 0 |
| B. With labels | 35 (71.4%) | 24 |
| C. Without labels | 35 (71.4%) | 34 |

| Compared with baseline | B | C |
| --- | --- | --- |
| Right before, wrong after | 3 | 3 |
| Wrong before, right after | 1 | 1 |
| Unchanged | 45 | 45 |
| Exact McNemar p | 0.63 | 0.63 |

Accuracy did not differ. Tracing the new errors showed two failure patterns:

- **Without labels, the Reflector doubted correct answers.** It wrote rules such as *"If a customer is unable to transfer funds, it may indicate that the transfer is still pending rather than failed"* from questions the model had answered correctly.
- **With labels, one ambiguous example became a general rule.** One build question was labelled `failed_transfer` although the customer described a rejection. The Reflector turned it into a rule that later misclassified two explicit "declined" questions.

Post-hoc check: removing the suspect bullet from B restored both questions (35 → 37). Removing two suspect bullets from C restored one of two (35 → 36). These bullets were chosen after seeing the results, so this supports the explanation for these cases only.

**Limitations:** small sample and a single run; failure cases were analysed individually; offline setting only; different task and model from the paper.

## Run it

```bash
pip install openai
export OPENROUTER_API_KEY=...        # any OpenRouter key
python3 ace_exp.py pilot             # optional sanity check on 21 train questions
python3 ace_exp.py traj
python3 ace_exp.py build
python3 ace_exp.py test
python3 ace_exp.py paired
python3 ace_exp.py ablate
```

Every model call is cached in `out/cache_*.jsonl`, so rerunning with this folder as-is reproduces the results above without new API calls. The full run costs about US$0.06. Set `ACE_MODEL` to try another model.

## Files

| Path | Contents |
| --- | --- |
| `ace_exp.py` | The whole experiment: data split, three roles, all stages |
| `data/` | Banking77 CSVs from the dataset's GitHub repository |
| `out/trajectories.json` | Generator answers on the 21 build questions |
| `out/build_log_B.json`, `out/build_log_C.json` | Every reflection and the bullets it added |
| `out/playbook_B.json`, `out/playbook_C.json` | Final playbooks, each bullet with its source question |
| `out/test_results.json` | Per-question predictions for A, B and C |
| `out/paired.json` | Per-question changes against the baseline |
| `out/ablation_results.json` | Predictions after removing the suspect bullets |
| `out/pilot.json`, `out/cache_*.jsonl` | Pilot answers; cached model calls (including two other models tried in the pilot) |

Banking77 is released by PolyAI under CC BY 4.0.
