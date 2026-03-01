# Math Training Datasets for SOTA Models (Feb 2026)

## Pretraining Corpora

| Dataset | Size | Description |
|---------|------|-------------|
| [DeepSeekMath Corpus](https://github.com/deepseek-ai/DeepSeek-Math) | 120B tokens | Math-focused tokens mined from CommonCrawl (EN + CN). Largest open math pretraining corpus |
| [Open-Web-Math](https://huggingface.co/datasets/open-web-math/open-web-math) | ~15B tokens | High-quality math web text filtered from CommonCrawl |
| [Nemotron-CC-Math](https://huggingface.co/blog/nvidia/nemotron-cc-math) | Large-scale | NVIDIA's web-scale math pretraining corpus, part of Nemotron-CC-v2 |

## SFT / Distillation Datasets

| Dataset | Size | Source model | Notes |
|---------|------|-------------|-------|
| [OpenMathInstruct-2](https://openreview.net/pdf?id=l5FDMofecw) | **14M** pairs | Llama 3.1 405B | NVIDIA. Key finding: question diversity > solution quantity. Verbose solutions hurt SFT |
| [Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | 347K problems, **7M** traces | gpt-oss-120b | 6 reasoning modes (high/med/low × with/without Python TIR). Verified via LLM-as-judge. Achieves **100% maj@16 on AIME 2024/2025** with Qwen3-8B |
| [NVIDIA Nemotron Post-Training v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | **30M** examples | Various | 20M math + 10M code + instruction following |
| [OpenMathInstruct-1](https://arxiv.org/abs/2402.10176) | 1.8M pairs | Mixtral | Permissive license. GSM8K + MATH problems |
| [SYNTHETIC-1](https://www.primeintellect.ai/blog/synthetic-1-release) | **2M** traces | DeepSeek-R1 | Prime Intellect. Math + code + science reasoning traces with verifiers |
| [NuminaMath](https://huggingface.co/AI-MO) | Large | GPT-4o assisted | Largest competition-level collection, but **restrictive license** due to GPT-4o usage |

## RL / RLVR Datasets

RLVR (RL from Verifiable Rewards) is the dominant paradigm now.

| Dataset | Size | Key property |
|---------|------|-------------|
| [Big-Math](https://arxiv.org/abs/2502.17387) | **250K+** problems | Stanford/SynthLabs. Curated for RL: open-ended, uniquely verifiable, closed-form answers. 10× larger than MATH |
| [Big-Math-Reformulated](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) | 47K additional | MCQs reformulated as open-ended for RL |
| [AoPS Dataset](https://openreview.net/forum?id=Bgz3okeZ7H) | 650K+ QA pairs | Olympiad-level problems from Art of Problem Solving forums |
| [DeepScaleR curriculum](https://www.emergentmind.com/topics/deepscaler) | Blended | AIME + AMC + MATH500 mix with curriculum-based RL |

## Benchmarks / Evaluation (not for training, but define the frontier)

- [MathArena](https://huggingface.co/datasets/MathArena/aime_2026_I) — live competition problems (AIME 2026, HMMT 2025, IMC 2025, APEX 2025)
- [OlymMATH](https://github.com/RUCAIBox/OlymMATH) — olympiad-level eval, 582K entries across 28 models
- [MATH-Vision](https://huggingface.co/datasets/MathLLMs/MathVision) — 3K problems with visual context

## Key Trends

1. **RLVR is king.** The dominant recipe is SFT warmup → RL with verifiable rewards. Datasets are increasingly designed with answer-verifiability as the primary constraint (Big-Math, Nemotron-Math-v2).

2. **Synthetic data quality problem.** Random sampling shows ~40% error rate in existing synthetic math datasets. Best practice is now multi-mode generation + verification pipelines (Nemotron-Math-v2's approach).

3. **Scale explosion.** From 1.8M (OpenMathInstruct-1, early 2024) → 14M (OpenMathInstruct-2) → 30M (Nemotron Post-Training v2) in under 2 years.

4. **Formal verification emerging.** Lean4-verified problems are increasingly seen as the gold standard, eliminating hallucination/ambiguity risks. [DeepSeekMath-V2](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2) uses self-verification and hits 118/120 on Putnam 2024.

5. **Correct-answer/wrong-proof gap.** A known failure mode of RLVR: models get right answers through flawed reasoning. Active research area — process reward models and proof verification are partial mitigations.
