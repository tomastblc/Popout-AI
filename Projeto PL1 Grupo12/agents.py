import random
from dataclasses import dataclass

from libs.mcts import MCTS, MCTSConfig


@dataclass(frozen=True)
class AgentSpec:
    name: str
    kind: str
    config: MCTSConfig | None = None
    random_seed: int | None = None


class BaseAgent:
    def choose_move(self, state):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, state):
        moves = state.legal_moves()
        if not moves:
            return None
        return self.rng.choice(moves)


class MCTSAgent(BaseAgent):
    def __init__(self, config: MCTSConfig):
        self.config = config
        self.engine = MCTS(config=config)

    def choose_move(self, state):
        return self.engine.search(state)


def build_agent(spec: AgentSpec):
    if spec.kind == "mcts":
        if spec.config is None:
            raise ValueError("MCTS agent requires a config")
        return MCTSAgent(spec.config)
    if spec.kind == "random":
        return RandomAgent(seed=spec.random_seed)
    raise ValueError(f"Unsupported agent kind: {spec.kind}")


def _with_iterations(spec: AgentSpec, iterations: int, suffix: str):
    if spec.kind != "mcts" or spec.config is None:
        return spec
    return AgentSpec(
        name=f"{spec.name}_{suffix}",
        kind=spec.kind,
        config=MCTSConfig(**{**spec.config.__dict__, "iterations": iterations}),
        random_seed=spec.random_seed,
    )


def _base_mcts_specs():
    return [
        AgentSpec(
            name="UCT_base",
            kind="mcts",
            config=MCTSConfig(iterations=500, exploration_c=1.414, rollout_policy="random"),
        ),
        AgentSpec(
            name="UCT_low_c",
            kind="mcts",
            config=MCTSConfig(iterations=500, exploration_c=0.7, rollout_policy="random"),
        ),
        AgentSpec(
            name="UCT_high_c",
            kind="mcts",
            config=MCTSConfig(iterations=500, exploration_c=2.0, rollout_policy="random"),
        ),
        AgentSpec(
            name="UCT_k3_random_expand",
            kind="mcts",
            config=MCTSConfig(
                iterations=500,
                exploration_c=1.414,
                expansion_policy="random_k",
                expansion_k=3,
                rollout_policy="random",
            ),
        ),
        AgentSpec(
            name="UCT_heuristic_rollout",
            kind="mcts",
            config=MCTSConfig(
                iterations=500,
                exploration_c=1.414,
                rollout_policy="heuristic",
            ),
        ),
        AgentSpec(
            name="UCT_epsilon_greedy",
            kind="mcts",
            config=MCTSConfig(
                iterations=500,
                exploration_c=1.414,
                rollout_policy="epsilon_greedy",
                epsilon=0.15,
            ),
        ),
        AgentSpec(
            name="UCT_depth_limited",
            kind="mcts",
            config=MCTSConfig(
                iterations=500,
                exploration_c=1.414,
                rollout_policy="heuristic",
                rollout_depth_limit=8,
            ),
        ),
    ]


def basic_mcts_specs():
    return _base_mcts_specs()


def up2k_mcts_specs():
    return [_with_iterations(spec, 2000, "2k") for spec in _base_mcts_specs()]


def up10k_mcts_specs():
    return [_with_iterations(spec, 10000, "10k") for spec in _base_mcts_specs()]


def default_mcts_specs():
    base_specs = _base_mcts_specs()
    specs_2000 = [_with_iterations(spec, 2000, "2k") for spec in base_specs]
    specs_10000 = [_with_iterations(spec, 10000, "10k") for spec in base_specs]
    return base_specs + specs_2000 + specs_10000
