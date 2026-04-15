# Cognition Engine

A cognitively-grounded agent for the [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) competition.

## Architecture

Four-stage cognitive cascade — no neural networks, no LLMs, pure algorithmic reasoning:

1. **Perception** — Connected-component segmentation + object tracking across frames
2. **World Model** — Symbolic rule induction from observed state transitions
3. **Planning** — Monte Carlo Tree Search (MCTS) with hierarchical priors and simulation ensemble
4. **Metacognition** — Adaptive strategy switching when the world model lacks traction

### Key Design Decisions

- **Hierarchical Prior**
  - Level 0: Core Knowledge module inspired by Spelke's Core Knowledge Framework
  - Level 1: empirical rules
  - Level 2: library transfer
- **Simulation Ensemble**: N MCTS trees x K sims, one per goal hypothesis, winner by root Q-value
- **Click Actions in MCTS**: the tree reasons about clicking specific objects, not just cardinal movement
- **Adaptive Exploration**: when MCTS Q-values are low, falls back to Explorer instead of wasting actions
- **Probe Optimization**: 30-sim probe before committing full simulations; early-exit when world model has no traction

## Dependencies

- `numpy`, `pillow`, `pydantic`
- `arc-agi`, `arcengine` (competition SDK)
- No PyTorch, no OpenAI, no internet access at runtime

## Usage

Requires the [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) toolkit:

```bash
# From the ARC-AGI-3-Agents directory:
uv run main.py --agent=cognitiveagent --game=<game_id>
```

## License

MIT License
