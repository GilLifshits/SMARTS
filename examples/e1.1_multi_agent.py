"""This multi-agent example in SMARTS showcases a simulation with diverse agent behaviors to illustrate the flexibility
 and complexity of multi-agent interactions. The script initializes a simulation environment with four different agents:
  RandomLanerAgent, KeepLaneAgent, and CautiousAgent.
  Each agent has a unique driving style—random lane changes, maintaining lanes, and cautious deceleration.
  The scenario runs multiple episodes where agents dynamically interact with the environment,
  demonstrating varied responses and behaviors in an intersection setting.
  The setup ensures a realistic and varied simulation,
  enhancing the understanding of multi-agent dynamics in autonomous driving"""
import random
import sys
from pathlib import Path
from typing import Final

import gymnasium as gym

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
sys.path.append('/home/amark/SMARTS')
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios

N_AGENTS = 4
AGENT_IDS: Final[list] = ["Agent %i" % i for i in range(N_AGENTS)]

class RandomLanerAgent(Agent):
    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def act(self, obs, **kwargs):
        return self._action_space.sample()

class KeepLaneAgent(Agent):
    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def act(self, obs, **kwargs):
        return self._action_space.sample()

class CautiousAgent(Agent):
    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def act(self, obs, **kwargs):
        action = self._action_space.sample()
        if isinstance(action, (list, tuple)):
            action[0] = max(action[0] - 0.3, -1.0)  # Decelerate more cautiously
        elif isinstance(action, (int, float)):
            action = max(action - 0.3, -1.0)
        return action

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_interfaces = {
        agent_id: AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        )
        for agent_id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=agent_interfaces,
        headless=headless,
    )

    agent_classes = [RandomLanerAgent, KeepLaneAgent, CautiousAgent, KeepLaneAgent]

    for episode in episodes(n=num_episodes):
        agents = {
            agent_id: agent_classes[i % len(agent_classes)](env.action_space[agent_id])
            for i, agent_id in enumerate(agent_interfaces.keys())
        }
        observations, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminateds = {"__all__": False}
        while not terminateds["__all__"]:
            actions = {
                agent_id: agent.act(observations) for agent_id, agent in agents.items()
            }
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode.record_step(observations, rewards, terminateds, truncateds, infos)

    env.close()

if __name__ == "__main__":
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "intersections" / "4lane_2.1"),
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
