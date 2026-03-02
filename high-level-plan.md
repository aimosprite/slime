HIGH LEVEL PLAN RIGHT NOW:
- check how easy it is to learn the embeddings 
- distill Qwen3-32B (we call it 32b in the future) into Qwen3-8b (we call it 8b in the future) (randomly initialize all the embeddings of the 8b model)
- randomly initialize the tokens of qwen3-8b. this experiment is to check can the model recover / learn the embeddings,
 how long it takes, and experiment with different initialization techniques. 

- they should both fit into the 8 gpu node. train qwen 8b using slime's sft and the script we'll provide.  

- use SLIME for sft training. 
- adopt tricks from rohin-experiments.md. but only what's relevant -- it was doing opd on two nodes, we are doing sft on one node

MAIN:
run DO_PREP=1 bash scripts/sft-qwen3-8b-AM-embedding-swap.sh. fix mistakes. 

so
- get me an SFT-ed version of the model. i trust you.
- log in to hugging face. checkpoint the model say, every 30 minutes. overwrite the latest checkpoints. 
- create a new hub in hf named "qwen3-32b-to-8b-embedding-surgery", or check if it exists
- and everything else that the script above does. look at the params used as well. 
- randomly initialize the embedding weights, and the last layer (so emb -> token should also be random)
- log useful stuff to WANDB. log a lot. 

TOKENS ARE AT .ENV

sft dataset:
- https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled


again, make a plan. but don't forget:
- checkpointing, hf
- wandb for evals
- we don't touch 32B for now. just SFT 8b on that sft dataset

IMPORTANT: 
- read CLAUDE.md, gimran-CLAUDE.md and MISTAKES.md
- when mistake happens, fix them. 
- USE EXISTING SCRIPTS, BUT CHECK THEIR CORRECTNESS. like sfacquire.sh, gimran-setup.sh, ...
- whenever you update something, push it to github. don't push model weights or other heavy stuff
- maintain MISTAKES.md, read from there to fix the mistakes you had. 