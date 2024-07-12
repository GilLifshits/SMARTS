from pathlib import Path  # Handle filesystem paths in an object-oriented way
from pprint import pprint as print  # Pretty-printing data structures
from typing import Dict, Literal, Optional, Union  # Provide type hints that enhance code readability and facilitate static type checking

import numpy as np

# Imports from Ray RLlib
try:
    from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.base_env import BaseEnv
    from ray.rllib.evaluation.episode import Episode
    from ray.rllib.evaluation.episode_v2 import EpisodeV2
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    from ray.rllib.policy.policy import Policy
    from ray.rllib.utils.typing import PolicyID
except Exception as e:
    from smarts.core.utils.custom_exceptions import RayException

    raise RayException.required_to("rllib.py")

# Imports from SMARTS
import smarts
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.sstudio.scenario_construction import build_scenarios

# Check if the script is being run directly and not imported as a module in another script
if __name__ == "__main__":
    from configs import gen_parser
    from rllib_agent import TrainingModel, rllib_agent
else:
    from .configs import gen_parser
    from .rllib_agent import TrainingModel, rllib_agent

# Define custom callbacks for handling events during training episodes.
# It tracks custom metrics (like the "ego" agent's rewards) and performs custom logging,
# enhancing the monitoring and evaluation of training progress.
# Add custom metrics to your tensorboard using these callbacks
# See: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):
        """
        Initializes a list for storing rewards at the start of each episode
        :param worker: The RolloutWorker instance managing the episode.
        :param base_env: The environment instance.
        :param policies: A dictionary mapping policy IDs to policy instances.
        :param episode: The current episode object.
        :param env_index: Index of the environment within a vectorized environment.
        :param kwargs: Additional keyword arguments.
        """
        episode.user_data["ego_reward"] = []

    @staticmethod
    def on_episode_step(
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):
        """
        Appends the current reward to the list for each step in the episode.
        :param worker: The RolloutWorker instance managing the episode.
        :param base_env: The environment instance.
        :param episode: The current episode object.
        :param env_index: Index of the environment within a vectorized environment.
        :param kwargs: Additional keyword arguments.
        """
        single_agent_id = list(episode.get_agents())[0]
        infos = episode._last_infos.get(single_agent_id)
        if infos is not None:
            episode.user_data["ego_reward"].append(infos["reward"])

    @staticmethod
    def on_episode_end(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):
        """
        Calculates and logs the mean reward for the episode and saves it in custom metrics.
        :param worker: The RolloutWorker instance managing the episode.
        :param base_env: The environment instance.
        :param policies: A dictionary mapping policy IDs to policy instances.
        :param episode: The current episode object.
        :param env_index: Index of the environment within a vectorized environment.
        :param kwargs: Additional keyword arguments.
        """
        mean_ego_speed = np.mean(episode.user_data["ego_reward"])
        print(
            f"ep. {episode.episode_id:<12} ended;"
            f" length={episode.length:<6}"
            f" mean_ego_reward={mean_ego_speed:.2f}"
        )
        episode.custom_metrics["mean_ego_reward"] = mean_ego_speed


def main(
    scenarios,
    envision,
    time_total_s,
    rollout_fragment_length,
    train_batch_size,
    seed,
    num_agents,
    num_workers,
    resume_training,
    result_dir,
    checkpoint_freq: int,
    checkpoint_num: Optional[int],
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
):
    """
    Serves as the entry point for running the RL training process.
    :param scenarios: The scenarios or environments used for training or evaluation.
    :param envision: A boolean indicating whether to enable visualization or not.
    :param time_total_s: The total time for training in seconds.
    :param rollout_fragment_length: The length of each rollout fragment.
    :param train_batch_size: The batch size used during training.
    :param seed: The seed used for random number generation, ensuring reproducibility.
    :param num_agents: The number of agents participating in the training.
    :param num_workers: The number of parallel workers used for training.
    :param resume_training: A boolean indicating whether to resume training from a checkpoint or start from scratch.
    :param result_dir: The directory where training results will be stored.
    :param checkpoint_freq: Specifies how often to save checkpoints during training.
    :param checkpoint_num: An optional parameter specifying the checkpoint number to resume training from.
    :param log_level: The logging level, with possible values being "DEBUG", "INFO", "WARN", or "ERROR".
    """
    rllib_policies = {
        f"AGENT-{i}": (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": TrainingModel.NAME}},
        )
        for i in range(num_agents)
    }
    agent_specs = {f"AGENT-{i}": rllib_agent["agent_spec"] for i in range(num_agents)}

    smarts.core.seed(seed)
    assert len(set(rllib_policies.keys()).difference(agent_specs)) == 0
    algo_config: AlgorithmConfig = (
        PPOConfig()
        .environment(
            env=RLlibHiWayEnv,
            env_config={
                "seed": seed,
                "scenarios": [
                    str(Path(scenario).expanduser().resolve().absolute())
                    for scenario in scenarios
                ],
                "headless": not envision,
                "agent_specs": agent_specs,
                "observation_options": "multi_agent",
            },
            disable_env_checking=True,
        )
        .framework(framework="tf2", eager_tracing=True)
        .rollouts(
            rollout_fragment_length=rollout_fragment_length,
            num_rollout_workers=num_workers,
            num_envs_per_worker=1,
            enable_tf1_exec_eagerly=True,
        )
        .training(
            lr_schedule=[[0, 1e-3], [1e3, 5e-4], [1e5, 1e-4], [1e7, 5e-5], [1e8, 1e-5]],
            train_batch_size=train_batch_size,
        )
        .multi_agent(
            policies=rllib_policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"{agent_id}",
        )
        .callbacks(callbacks_class=Callbacks)
        .debugging(log_level=log_level)
    )

    def get_checkpoint_dir(num):
        """
        Returns the checkpoint directory path based on the checkpoint number.
        :param num: The checkpoint number.
        """
        checkpoint_dir = Path(result_dir) / f"checkpoint_{num}" / f"checkpoint-{num}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    if resume_training:
        checkpoint = str(get_checkpoint_dir("latest"))
    if checkpoint_num:
        checkpoint = str(get_checkpoint_dir(checkpoint_num))
    else:
        checkpoint = None

    print(f"======= Checkpointing at {str(result_dir)} =======")

    algo = algo_config.build()
    if checkpoint is not None:
        algo.load_checkpoint(checkpoint=checkpoint)
    result = {}
    current_iteration = 0
    checkpoint_iteration = checkpoint_num or 0

    try:
        while result.get("time_total_s", 0) < time_total_s:
            result = algo.train()
            print(f"======== Iteration {result['training_iteration']} ========")
            print(result, depth=1)

            if current_iteration % checkpoint_freq == 0:
                checkpoint_dir = get_checkpoint_dir(checkpoint_iteration)
                print(f"======= Saving checkpoint {checkpoint_iteration} =======")
                algo.save_checkpoint(checkpoint_dir)
                checkpoint_iteration += 1
            current_iteration += 1
        algo.save_checkpoint(get_checkpoint_dir(checkpoint_iteration))
    finally:
        algo.save_checkpoint(get_checkpoint_dir("latest"))
        algo.stop()


# This block of code sets up the script's environment, parses command-line arguments, sets default values if necessary,
# builds scenarios, and finally calls the main function with the required arguments to start the RL training process.
if __name__ == "__main__":
    default_result_dir = str(Path(__file__).resolve().parent / "results" / "pg_results")
    parser = gen_parser("rllib-example", default_result_dir)
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=None,
        help="The checkpoint number to restart from.",
    )
    parser.add_argument(
        "--rollout_fragment_length",
        type=str,
        default="auto",
        help="Episodes are divided into fragments of this many steps for each rollout. In this example this will be ensured to be `1=<rollout_fragment_length<=train_batch_size`",
    )
    args = parser.parse_args()
    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop"),
        ]
    build_scenarios(scenarios=args.scenarios, clean=False, seed=args.seed)

    main(
        scenarios=args.scenarios,
        envision=args.envision,
        time_total_s=args.time_total_s,
        rollout_fragment_length=args.rollout_fragment_length,
        train_batch_size=args.train_batch_size,
        seed=args.seed,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        resume_training=args.resume_training,
        result_dir=args.result_dir,
        checkpoint_freq=max(args.checkpoint_freq, 1),
        checkpoint_num=args.checkpoint_num,
        log_level=args.log_level,
    )
