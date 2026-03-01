end goal: 
- distill qwen 3.5 122b into gpt oss 
- plan: 1) initialize the tokenizers, 2) sft, 3) opd
- need: 1) random / great initizalization 2) a lot of rollouts from the teacher, 3) opd pipeline (rohin), 4) all the other evals in the world. this is quite a surgery 
mid term goal:
- distill qwen 3.5 122b-10a into 32b-3a by initializing tokenizers



RIGHT NOW:
- distill Qwen3-32B  into Qwen3-8b
- randomly initialize the tokens of qwen3-8b. this experiment is to check can the model recover / learn the embeddings, how long it takes, and experiment with different initialization techniques. 
- generate a bunch of rollouts of the teacher. first experiment with math data only. i need a huuuge dataset. or we can just try on the dataset we have with just a bunch of epochs. 

future: 
- all three tokenizers of 3, 3.5 and oss are differnet. experiment with them, especially the tool use. 
- experiment with different tokenizer initializations than just random. 

why: 
- build a storng pipeline so that when deepseek v4 drops, we're ready 

questions: 
- what happens if you only train it on math? 
- tool use bench? instruction following bench? 
- should i use Qwen3-30B-A3B or 32b? 30B should be fine 
- make 30B rollouts
- compare math only SFT vs general SFT, and then math 
- 

other notes: 
- what are the quantization levels? 


what data to generate the rollouts on? 
- 

v0: 
- delete the tokenizer of qwen 3 8b, and distill qwen 3 30b into it. 
- is sft needed? surely it is. 

sft: 
- on the rollouts of the teacher model, pre-generated. 
- in the future, take the 

- 

take
qwen

benchmarks we're aiming for:

|                   | Qwen3-8B Base | Qwen3-30B-A3B Base | Qwen3-32B Base |
|-------------------|---------------|---------------------|----------------|
| Architecture      | Dense         | MoE                 | Dense          |
| # Total Params    | 8B            | 30B                 | 32B            |
| # Activated Params| 8B            | 3B                  | 32B            |
| **General Tasks** |               |                     |                |
| MMLU              | 76.89         | 81.38               | 83.61          |
| MMLU-Redux        | 76.17         | 81.17               | 83.41          |
| MMLU-Pro          | 56.73         | 61.49               | 65.54          |
| SuperGPQA         | 31.64         | 35.72               | 39.78          |
| BBH               | 78.40         | 81.54               | 87.38          |
| **Math & STEM**   |               |                     |                |
| GPQA              | 44.44         | 43.94               | 49.49          |
| GSM8K             | 89.84         | 91.81               | 93.40          |
| MATH              | 60.80         | 59.04               | 61.62          |
| **Coding Tasks**  |               |                     |                |
| EvalPlus          | 67.65         | 71.45               | 72.05          |
| MultiPL-E         | 58.75         | 66.53               | 67.06          |
| MBPP              | 69.80         | 74.40               | 78.20          |
| CRUX-O            | 62.00         | 67.20               | 72.50          |
| **Multilingual**  |               |                     |                |
| MGSM              | 76.02         | 79.11               | 83.06          |
| MMMLU             | 75.72         | 81.46               | 83.83          |
| IINCLUDE          | 59.40         | 67.00               | 67.87          |