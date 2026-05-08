# SOVEREIGN — RL Final Project

This is my final project for the RL class. The assignment gave us a custom game called SOVEREIGN and asked us to build it as a Gymnasium environment, train a PPO agent on it, and run a bunch of ablation experiments. The game is basically a stylized geopolitical conflict — you play as an "Invader" nation that has twice the military units as the Defender, and the whole point is to see whether the agent figures out on its own that invading is actually a bad strategy once you account for legitimacy loss, occupation costs, and international pressure.

## What the game is

There are 9 territories laid out as a graph. I0 is the Invader home, D0 is the Defender home, N0 is the Neutral nation's home, and the rest are contested territories in between. The Invader starts with 15 units at I0, Defender has 7 at D0.

Every step the agent picks two things at once — a political action and a military action. Political actions are things like NEGOTIATE, SEEK_ALLIANCE, IMPOSE_SANCTION etc. Military actions are ADVANCE, HOLD, WITHDRAW, STRIKE. The combination of what you do every step affects three key variables: legitimacy (L), neutral posture (theta), and occupation time (t_occ). Those three things feed into the reward function in a way that makes aggressive play increasingly expensive over time.

The Defender is rule-based — it prioritizes retaking its home territory, then recapturing C1/C3 if invaded, then reinforcing when threatened. It doesn't learn anything, it just reacts.

The state is a flat 49-dim vector — 27 values for territory control (one-hot per territory), 9 for invader unit counts, 9 for defender unit counts, and 4 scalars (L, E, theta, t_occ).

## How to run it

Install dependencies first:

```
pip install gymnasium stable-baselines3 networkx matplotlib
```

Note: this crashed on Python 3.13 with PyTorch 2.9 due to a known incompatibility. I had to use a Python 3.10 environment. If you hit a segfault on import, that's probably why.

To train everything (full model with 3 seeds, all 5 ablations, sanction threshold sweep, non-learning baselines):

```
python train.py
```

Takes a while since it's 500k timesteps per model and there are 9+ of them. Models and reward logs get saved to `models/`.

To evaluate and generate plots:

```
python evaluate.py
```

Plots go to `plots/`.

## Ablation experiments

The environment has three flags you can toggle: `use_legitimacy`, `use_occupation_cost`, `use_neutral_posture`. The idea is to turn each one off and see how the agent's behavior changes. There's also a baseline condition where all three are off.

There's a fourth experiment where we sweep the sanction threshold (0.40, 0.60, 0.80) to see how early sanctions kick in affects what the agent learns.

## What actually happened with the results

The full model agent ended up using NEGOTIATE about 72% of the time and HOLDing most of the time militarily. So it did learn that diplomacy was better than fighting, which is what the assignment was trying to show.

The more surprising result was the `no_occ_cost` and `baseline` conditions. I expected the agent to just invade hard when there's no occupation penalty, but it actually converged to sitting at home and HOLDing almost every step (~98% HOLD). Turns out if there's no penalty for doing nothing, the best strategy is to just collect your home territory's resource value every step for 200 steps and never risk losing units. The agent found that on its own which I didn't expect.

The `no_legitimacy` condition switched to heavy STRIKE usage (86%) which makes more sense — if legitimacy doesn't matter, why not just destroy Defender units.

The sanction threshold sweep didn't show a huge difference between 0.40, 0.60, and 0.80 in terms of final reward (~63-71 range). The 0.40 threshold did converge slightly higher which aligns with the idea that earlier sanctions force the agent to learn diplomacy sooner.

## Files

`sovereign_env.py` — the whole environment. Graph setup, all action logic, defender AI, combat, reward function, ablation flags.

`train.py` — runs all experiments, logs rewards with a callback, saves everything to models/.

`evaluate.py` — loads models, runs 100 eval episodes per condition, prints dominant action analysis, saves all plots.

## Limitations

The Defender has no political actions which probably underestimates how much diplomatic pressure a real opponent would apply. The Neutral nation is stochastic but doesn't learn anything either. And the settlement condition requires t_occ == 0 which means the agent basically has to fully withdraw before it can negotiate — that made the +40 settlement reward pretty hard to reach in practice so the agent mostly ended up earning rewards through sustained negotiation rather than actually triggering settlement.
