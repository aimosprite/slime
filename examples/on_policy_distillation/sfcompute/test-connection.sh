#!/bin/bash
set -euo pipefail

# Quick connection test between two nodes using Docker + Ray (same as training setup).
# Usage:
#   Teacher/head node:  bash examples/on_policy_distillation/sfcompute/test-connection.sh head
#   Student/worker node: bash examples/on_policy_distillation/sfcompute/test-connection.sh worker <head_ip>
#
# Tests: Tailscale connectivity, Docker --network host, Ray cluster formation, GPU visibility.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE="${IMAGE:-slimerl/slime:latest}"
RAY_PORT="${RAY_PORT:-6379}"
MODE="${1:-}"
HEAD_IP="${2:-}"

usage() {
    cat <<'EOF'
Usage:
  bash examples/on_policy_distillation/sfcompute/test-connection.sh head
  bash examples/on_policy_distillation/sfcompute/test-connection.sh worker <head_ip>

Runs a quick Ray cluster test inside Docker containers (same networking as training).
EOF
}

detect_primary_ip() {
    if command -v tailscale >/dev/null 2>&1; then
        local ip
        ip="$(tailscale ip -4 2>/dev/null | awk 'NF {print; exit}' || true)"
        if [ -n "${ip}" ]; then
            echo "${ip}"
            return 0
        fi
    fi
    python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
    s.close()
except Exception:
    print('127.0.0.1')
" 2>/dev/null || echo "127.0.0.1"
}

case "${MODE}" in
    head)
        HEAD_IP="${HEAD_IP:-$(detect_primary_ip)}"
        echo "=== Connection Test: HEAD mode ==="
        echo "  Image:   ${IMAGE}"
        echo "  Head IP: ${HEAD_IP}"
        echo "  Port:    ${RAY_PORT}"
        echo ""
        echo "Run this on the worker/student node:"
        echo "  RAY_PORT=${RAY_PORT} bash examples/on_policy_distillation/sfcompute/test-connection.sh worker ${HEAD_IP}"
        echo ""

        docker run --rm --gpus all --network host --ipc=host \
            -e HEAD_IP="${HEAD_IP}" \
            -e RAY_PORT="${RAY_PORT}" \
            "${IMAGE}" \
            bash -lc '
set -euo pipefail
echo "--- Inside Docker container (head) ---"
echo "1. Checking network interfaces for ${HEAD_IP}..."
if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((\"${HEAD_IP}\", 0))
    s.close()
    print(\"  OK: Can bind to ${HEAD_IP}\")
except OSError as e:
    print(f\"  FAIL: Cannot bind to ${HEAD_IP}: {e}\")
    sys.exit(1)
"; then
    echo ""
else
    echo "  This means the Tailscale interface is NOT visible inside Docker."
    echo "  Check: tailscale is using kernel networking (not userspace)"
    exit 1
fi

echo "2. Starting Ray head on ${HEAD_IP}:${RAY_PORT}..."
ray stop --force 2>/dev/null || true
ray start --head --node-ip-address "${HEAD_IP}" --num-gpus 0 --port "${RAY_PORT}" --disable-usage-stats

echo ""
echo "3. Ray head started. Checking cluster..."
python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
res = ray.cluster_resources()
print(f\"  Cluster resources: {dict(res)}\")
print(f\"  GPUs on head: {int(res.get(\"GPU\", 0))}\")
ray.shutdown()
"

echo ""
echo "4. Waiting for worker to join (up to 5 min)..."
echo "   Start the worker node now if you have not already."
for i in $(seq 1 20); do
    gpu_count=$(python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
print(int(ray.cluster_resources().get(\"GPU\", 0)))
ray.shutdown()
" 2>/dev/null || echo 0)
    if [ "${gpu_count}" -gt 0 ]; then
        echo ""
        echo "=== SUCCESS: Worker connected! ${gpu_count} GPUs in cluster ==="
        python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
nodes = ray.nodes()
for n in nodes:
    alive = n.get(\"Alive\", False)
    ip = n.get(\"NodeManagerAddress\", \"?\")
    res = n.get(\"Resources\", {})
    gpus = int(res.get(\"GPU\", 0))
    print(f\"  Node {ip}: alive={alive}, GPUs={gpus}\")
ray.shutdown()
"
        echo ""
        echo "=== All tests passed! Training should work. ==="
        ray stop --force 2>/dev/null || true
        exit 0
    fi
    echo "  ${gpu_count} GPUs so far (${i}/20, waiting 15s)..."
    sleep 15
done
echo "TIMEOUT: No worker GPUs joined after 5 min."
ray stop --force 2>/dev/null || true
exit 1
'
        ;;

    worker)
        if [ -z "${HEAD_IP}" ]; then
            echo "Error: head IP required. Usage: $0 worker <head_ip>"
            exit 1
        fi
        echo "=== Connection Test: WORKER mode ==="
        echo "  Image:   ${IMAGE}"
        echo "  Head IP: ${HEAD_IP}"
        echo "  Port:    ${RAY_PORT}"
        echo ""

        # Detect GPU count on this host
        GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || echo 8)"
        echo "  GPUs:    ${GPU_COUNT}"
        echo ""

        docker run --rm --gpus all --network host --ipc=host \
            -e HEAD_IP="${HEAD_IP}" \
            -e RAY_PORT="${RAY_PORT}" \
            -e GPU_COUNT="${GPU_COUNT}" \
            "${IMAGE}" \
            bash -lc '
set -euo pipefail
echo "--- Inside Docker container (worker) ---"

echo "1. Testing TCP connectivity to ${HEAD_IP}:${RAY_PORT}..."
if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)
try:
    s.connect((\"${HEAD_IP}\", int(\"${RAY_PORT}\")))
    s.close()
    print(\"  OK: TCP connection to ${HEAD_IP}:${RAY_PORT} succeeded\")
except Exception as e:
    print(f\"  FAIL: Cannot reach ${HEAD_IP}:${RAY_PORT}: {e}\")
    print(\"  Make sure the head node test is running first.\")
    sys.exit(1)
"; then
    echo ""
else
    exit 1
fi

echo "2. Joining Ray cluster at ${HEAD_IP}:${RAY_PORT} with ${GPU_COUNT} GPUs..."
ray stop --force 2>/dev/null || true
for attempt in $(seq 1 10); do
    if ray start --address "${HEAD_IP}:${RAY_PORT}" --num-gpus "${GPU_COUNT}" --disable-usage-stats; then
        echo "  OK: Joined Ray cluster"
        break
    fi
    echo "  Attempt ${attempt}/10 failed, retrying in 5s..."
    ray stop --force 2>/dev/null || true
    sleep 5
done

echo ""
echo "3. Verifying cluster membership..."
python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
nodes = ray.nodes()
total_gpus = 0
for n in nodes:
    ip = n.get(\"NodeManagerAddress\", \"?\")
    res = n.get(\"Resources\", {})
    gpus = int(res.get(\"GPU\", 0))
    total_gpus += gpus
    print(f\"  Node {ip}: GPUs={gpus}\")
print(f\"  Total cluster GPUs: {total_gpus}\")
ray.shutdown()
"
echo ""
echo "=== Worker test done. Check the head node for final result. ==="
echo "Keeping worker alive for 30s so head can verify..."
sleep 30
ray stop --force 2>/dev/null || true
'
        ;;

    *)
        usage
        exit 1
        ;;
esac
