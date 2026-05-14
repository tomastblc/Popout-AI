import random
from dataclasses import dataclass

from board import Move
from mcts import MCTS, MCTSConfig

"""
--------------------------------------------------------
MCTS Configuration Documentation
--------------------------------------------------------
Monte Carlo Tree Search (MCTS) is a search algorithm
based on building a search tree and evaluating positions
through repeated simulated playthroughs (rollouts).

--------------------------------------------------------
exploration_c
--------------------------------------------------------

Type:
    float

Description:
    Exploration coefficient used in the UCT formula.

    Controls the balance between:
    - exploitation (using already strong moves)
    - exploration (testing less-visited moves)

Behavior:
    Larger values:
        - increase exploration
        - favor less-visited nodes

    Smaller values:
        - increase greediness
        - favor high-performing nodes

Typical values:
    0.5 - 2.0

Used in:
    child_selection = uct

--------------------------------------------------------
expansion_policy
--------------------------------------------------------

Type:
    enum

Possible values:
    all
    random_k
    heuristic_k

Description:
    Determines how nodes are expanded in the search tree.

Values:

    all
        Expand all legal child moves.

    random_k
        Expand K randomly selected moves.

    heuristic_k
        Expand the K best moves according
        to a heuristic evaluation.

Notes:
    - all creates a wider and more complete tree,
      but uses more memory.
    - heuristic_k is usually faster in large
      search spaces.

--------------------------------------------------------
expansion_k
--------------------------------------------------------

Type:
    integer

Description:
    Number of child nodes to expand when using:
        - expansion_policy = random_k
        - expansion_policy = heuristic_k

Example:
    expansion_policy = heuristic_k
    expansion_k = 5

    → expands the 5 best heuristic moves.

--------------------------------------------------------
rollout_policy
--------------------------------------------------------

Type:
    enum

Possible values:
    random
    heuristic
    epsilon_greedy

Description:
    Determines how rollout simulations are performed.

Values:

    random
        Fully random move selection.

    heuristic
        Moves selected using a heuristic policy.

    epsilon_greedy
        Usually selects the best move,
        but occasionally chooses a random move.

Notes:
    - random rollouts provide diversity.
    - heuristic rollouts are usually stronger,
      but may introduce bias.

--------------------------------------------------------
rollout_depth_limit
--------------------------------------------------------

Type:
    integer

Description:
    Maximum depth of a rollout simulation.

Behavior:
    If the game does not terminate before
    the limit is reached, the current position
    is evaluated heuristically.

Impact:

    Smaller values:
        + faster simulations
        - lower accuracy

    Larger values:
        + more realistic simulations
        - higher computational cost

--------------------------------------------------------
child_selection
--------------------------------------------------------

Type:
    enum

Possible values:
    uct
    greedy
    softmax

Description:
    Determines how child nodes are selected
    during tree traversal.

Values:

    uct
        Uses the Upper Confidence Bound (UCT) formula.

        Balances:
            - exploration
            - exploitation

    greedy
        Always selects the child with
        the highest current score.

    softmax
        Probabilistic selection:
        stronger nodes are selected more often,
        but weaker nodes still have a chance.

--------------------------------------------------------
draw_value
--------------------------------------------------------

Type:
    float

Description:
    Numerical value assigned to a draw result.

Typical values:

    0.0
        Draws are treated as undesirable.

    0.5
        Draws are treated as half a win.

    1.0
        Draws are treated as successful outcomes.

Example standard scoring:
    win  = 1.0
    draw = 0.5
    loss = 0.0

--------------------------------------------------------
epsilon
--------------------------------------------------------

Type:
    float

Description:
    Probability of selecting a random move
    in epsilon-greedy rollout policy.

Used in:
    rollout_policy = epsilon_greedy

Example:
    epsilon = 0.1

    → 10% random actions
    → 90% greedy actions

Typical values:
    0.05 - 0.2

--------------------------------------------------------
random_seed
--------------------------------------------------------

Type:
    integer

Description:
    Seed value for the random number generator.

Purpose:
    Makes MCTS behavior reproducible.

Using the same seed results in:
    - identical rollouts
    - identical search behavior
    - reproducible experiments and debugging

Example:
    random_seed = 42

--------------------------------------------------------
Recommended Defaults
--------------------------------------------------------

General-purpose MCTS configuration:

    exploration_c       = 1.4
    expansion_policy    = heuristic_k
    expansion_k         = 5
    rollout_policy      = epsilon_greedy
    rollout_depth_limit = 50
    child_selection     = uct
    draw_value          = 0.5
    epsilon             = 0.1
    random_seed         = 42

--------------------------------------------------------
"""

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
