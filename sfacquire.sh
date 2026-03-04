#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STARTUP_SCRIPT="$SCRIPT_DIR/pass_along_node_to_setup.sh"

if [ ! -f "$STARTUP_SCRIPT" ]; then
    echo "ERROR: pass_along_node_to_setup.sh not found at $STARTUP_SCRIPT"
    echo ""
    echo "Create pass_along_node_to_setup.sh with your SSH key so VMs accept your connection:"
    echo ""
    echo '  #!/bin/bash'
    echo '  mkdir -p /root/.ssh'
    echo '  cat >>/root/.ssh/authorized_keys <<"EOF"'
    echo '  ssh-ed25519 AAAA... your-public-key-here'
    echo '  EOF'
    echo ""
    echo "pass_along_node_to_setup.sh is gitignored (contains keys). See aimo26/pass_along_node_to_setup.sh for reference."
    exit 1
fi

# Show zones and pick one
echo "Fetching available zones..."
echo ""
sf zones ls
echo ""
echo "Pick a zone from the list above."
read -p "Zone [richmond]: " zone
zone=${zone:-richmond}

# Count
read -p "Number of nodes [1]: " count
count=${count:-1}

# Duration
read -p "Duration (e.g. 1h, 2d, 30m) [1h]: " duration
duration=${duration:-1h}
# If user typed a bare number, append 'h'
if [[ "$duration" =~ ^[0-9]+$ ]]; then
    duration="${duration}h"
fi

# Price
read -p "Max price $/node-hour [12.00]: " price
price=${price:-12.00}

# Name
read -p "Node name [slime]: " name
name=${name:-slime}

echo ""
echo "sf nodes create $name -n $count -z $zone --duration $duration -p $price --user-data-file $STARTUP_SCRIPT"
echo ""
read -p "Run? [Y/n] " confirm
confirm=${confirm:-Y}
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

sf nodes create "$name" \
    -n "$count" \
    -z "$zone" \
    --duration "$duration" \
    -p "$price" \
    --user-data-file "$STARTUP_SCRIPT"
