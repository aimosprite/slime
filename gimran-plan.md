MAIN PLAN: 

plan: swap out gpt oss's (20b) tokenizer to qwen3.5 tokenizer, and SFT oss on qwen3.5's rollouts. evaluate how good this 'surgery' trick is (on math datasets)
- study how we did qwen3 SFT before. do like that rn. wandb logging, hf checkpoints, etc. one mistake: we added a model as lerchen3/...., but it should be inside aimosprite/ hub in hf. move the previous model from lerchen3 to aimosprite, and make sure new model get's checkpointed to aimosprite/ hf hub. 
- the sft script is scripts/sft-qwen3-8b-AM-embedding-swap.sh. make a config file. everything there was super important. don't miss anything. 
- qwen 3.5 rollouts are this: https://huggingface.co/datasets/aimosprite/qwen3.5-35b-eval-run-20260302)
- sanity check: compare the sizes of the datasets (previous vs current at https://huggingface.co/datasets/aimosprite/qwen3.5-35b-eval-run-20260302). just so i know
- take gpt oss, and swap out its tokenizer to the tokenizer of qwen3.5. download the official tokenizer of 3.5, double check it's official. it's probably there: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- so gpt oss first layer (tokens -> emb), and last layer (emb -> tokens) now change, because it now has a new tokenizer. randomly initialize all the embeddings. double-check that you swapped out the tokenizer of gpt oss 20b to qwen3.5's tokenizer everywhere. 
- freeze all the weights of gpt oss 20b except the first and last layer (new layers, randomly initialized, new embeddings and new tokenizer)
- run sft on qwen3.5's rollouts. 
- as before, first things first generate the expected sft loss (of oss without the tokenizer swap) to see where should we aim. 
- estimate how long it will take for the sft to finish. 
- look up the benchmarks of gpt-oss-20b (without tool calls) on math datasets
- btw, everything you do today with gpt-oss-20b is without tool calls. just ignore tool calls for now. 

PLAN: 
- first "pre-train" on https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled until the test loss is nearby the test loss for gpt oss 20b without the tokenizer trick
- then train on the https://huggingface.co/datasets/aimosprite/qwen3.5-35b-eval-run-20260302

IMPORTANT:
- have test and train sets for both datasets we use 
- TOKENS ARE AT .ENV
- sft/3.5 rollouts dataset: https://huggingface.co/datasets/aimosprite/qwen3.5-35b-eval-run-20260302) 
- copy scripts/sft-qwen3-8b-AM-embedding-swap.sh almost line-by-line. make a config file. 
- read CLAUDE.md, gimran-CLAUDE.md and MISTAKES.md
- when mistake happens, fix them. 
- USE EXISTING SCRIPTS, BUT CHECK THEIR CORRECTNESS. like sfacquire.sh, gimran-setup.sh, ...
- whenever you update something, push it to github. don't push model weights or other heavy stuff
- maintain MISTAKES.md, read from there to fix the mistakes you had. 
- check of OOMs. 
- whenever you update something, add it to github. token is at .env. don't add heavy files (like model weights etc). IMPORTANT
- TEST EVERYTHING. THE LAST THING I WANT IS TO RUN FOR 5 HOURS JUST TO DETERMINE THE TOKENIZER IS WRONG. TOKENIZERS CAN HAVE VERY VERY SUBTLE MISTAKES. DON'T TRUST YOUR JUDGEMENT, DO NOT SILENTLY DEFAULT. ESPECIALLY LOOK UP TOKENIZERS PLZ. AND SPECIAL TOKENS. THIS IS THE MOST IMPORTANT PART.

again, make a plan. but don't forget:
- checkpointing, hf
- wandb for evals


EVERYTHING I SAID IS SUPER IMPORTANT. DON'T MISS ANYTHING ^^^

---------------------------------------------------------------