FROM slimerl/slime:latest

# ====================================== Gimran tooling ============================================
# Everything below is gimran-specific (Claude Code, gh CLI, etc.)
# The base image already has: CUDA, PyTorch, flash-attn, Megatron (patched),
# sglang (patched), apex, transformer_engine, slime — all pre-compiled.

ARG DEBIAN_FRONTEND=noninteractive

# ── Node.js 22 + Claude Code ──────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    rm -rf /var/lib/apt/lists/*

# ── uv (fast Python package manager) ─────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ── GitHub CLI ────────────────────────────────────────────────────
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# ── hf CLI ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# ── Non-root user for sandboxed Claude execution ─────────────────
RUN useradd -m -s /bin/bash aimo && \
    echo "aimo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ── clauded wrapper (runs Claude as aimo user) ───────────────────
COPY gimran/clauded /usr/local/bin/clauded
RUN chmod +x /usr/local/bin/clauded

# ── Init script (first-boot: secrets, git config, models) ────────
COPY gimran/init.sh /usr/local/bin/gimran-init
RUN chmod +x /usr/local/bin/gimran-init

WORKDIR /root/slime
