# CLAUDE.md

Read README.md for a general structure of the repo.

## Rules

- When starting a big run (eval, rl, opd, ...), tell the user to make sure they made checkpoints and log everything.
- Follow the philosophy of "no silent defaults". Every config field that can be set must be explicitly set by humans. The eval script enforces this via `validate_config()`.
- When the user asks you to download models, ask for the model name and link at huggingface and download via hf to the `models/` folder (.gitignored).
- Log every mistake you made into MISTAKES.md. When running into problems, consult MISTAKES.md for help.
- Update the README.md after you made crucial changes.
- we use uv
- make sure any work you do is saved for the future -- ie when a user will boot up a new VM, they shouldn't run into the problems they had 

Downloading models:
```bash
hf download <model_name> --local-dir models/<model_name>
``` 
When you downloaded a model, add the info about it to models.txt