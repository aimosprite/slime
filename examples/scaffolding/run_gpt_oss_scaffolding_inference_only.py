import os

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking


def inference_only(args):
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS"), ignore_reinit_error=True, include_dashboard=False)

    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])

    start_rollout_id = args.start_rollout_id or 0
    for rollout_id in range(start_rollout_id, args.num_rollout):
        ray.get(rollout_manager.generate.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    inference_only(args)
