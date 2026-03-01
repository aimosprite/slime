HIGH LEVEL PLAN RIGHT NOW:
- check how easy it is to learn the embeddings 
- distill Qwen3-32B (we call it 32b in the future) into Qwen3-8b (we call it 8b in the future) (randomly initialize all the embeddings of the 8b model)
- randomly initialize the tokens of qwen3-8b. this experiment is to check can the model recover / learn the embeddings,
 how long it takes, and experiment with different initialization techniques. 
- generate a bunch of rollouts of the teacher. first experiment with math data only. 

- they should both fit into the 8 gpu node. train qwen 8b using megatron's sft. 
- randomly sample data from the openmathinstruct 32b model given 

- use SLIME for sft training. 
- adopt tricks from rohin-experiments.md. but only what's relevant -- it was doing opd on two nodes, we are doing sft on one node

set up: 
- look at scripts/run-qwen3-4B-base-sft.sh. 
so
- log in to hugging face.
- create a new hub in hf named "qwen3-32b-to-8b-embedding-surgery"
- checkpoint the 8b model there every 10 minutes with evals
- and everything else that the script above does. look at the params used as well. before you run, check everything with me


sft dataset:
- https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled


again, make a plan. but don't forget:
- checkpointing, hf
- wandb for evals
- we don't touch 32B for now. just SFT 8b on that sft dataset


