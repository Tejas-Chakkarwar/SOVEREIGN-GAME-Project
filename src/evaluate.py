import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from stable_baselines3 import PPO
from sovereign_env import SovereignEnv

# I'll make a plots directory to save everything
os.makedirs("plots", exist_ok=True)

# node index names — useful for the map visualization
NODE_NAMES = {
    0: "I0",
    1: "D0",
    2: "N0",
    3: "C0",
    4: "C1",
    5: "C2",
    6: "C3",
    7: "C4",
    8: "C5",
}

# action name maps — for printing dominant policy
POL_ACTION_NAMES = {
    0: "SEEK_ALLIANCE",
    1: "IMPOSE_SANCTION",
    2: "ISSUE_THREAT",
    3: "NEGOTIATE",
    4: "DO_NOTHING",
}

MIL_ACTION_NAMES = {
    0: "ADVANCE",
    1: "HOLD",
    2: "WITHDRAW",
    3: "STRIKE",
}


# =====================================================================
# Helper: evaluate a saved model for N episodes
# =====================================================================
def evaluate_model(model_path, env_kwargs, n_episodes=100):
    # load the model and run episodes, collect rewards and action counts
    env = SovereignEnv(**env_kwargs)
    model = PPO.load(model_path, env=env)

    all_rewards = []
    pol_action_counts = np.zeros(5)
    mil_action_counts = np.zeros(4)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            a_pol = int(action[0])
            a_mil = int(action[1])
            pol_action_counts[a_pol] += 1
            mil_action_counts[a_mil] += 1
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward

        all_rewards.append(ep_reward)

    return all_rewards, pol_action_counts, mil_action_counts


# =====================================================================
# Plot 1: Training reward curves (mean ± std) for full model seeds
# =====================================================================
def plot_full_model_curves():
    with open("models/full_model_rewards.pkl", "rb") as f:
        full_model_rewards = pickle.load(f)

    # I need to smooth and align the reward curves across seeds
    # they might have different lengths, so I'll interpolate to a common x-axis
    # actually I'll just use moving average and plot up to the shortest length

    seeds = [42, 123, 7]
    window = 20  # moving average window — helps smooth out the noise

    # find minimum length across seeds
    min_len = min(len(full_model_rewards[s]) for s in seeds)

    smoothed = {}
    for seed in seeds:
        rewards = full_model_rewards[seed][:min_len]
        # compute moving average
        smoothed_rewards = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed_rewards.append(np.mean(rewards[start:i+1]))
        smoothed[seed] = np.array(smoothed_rewards)

    # stack and compute mean/std across seeds
    all_smoothed = np.stack([smoothed[s] for s in seeds])
    mean_curve = np.mean(all_smoothed, axis=0)
    std_curve = np.std(all_smoothed, axis=0)

    x = np.arange(len(mean_curve))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean_curve, label="Full Model (mean across seeds)", color="steelblue")
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3, color="steelblue", label="±1 std")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Full Model Training Curves — Mean ± Std (3 Seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/full_model_curves.png", dpi=150)
    plt.close()
    print("Saved: plots/full_model_curves.png")


# =====================================================================
# Plot 2: Ablation training curves — all on one plot
# =====================================================================
def plot_ablation_curves():
    with open("models/ablation_rewards.pkl", "rb") as f:
        ablation_rewards = pickle.load(f)

    window = 20
    colors = {
        "full": "steelblue",
        "no_legitimacy": "tomato",
        "no_occ_cost": "seagreen",
        "no_neutral": "orange",
        "baseline": "gray",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rewards in ablation_rewards.items():
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))
        smoothed = np.array(smoothed)
        ax.plot(smoothed, label=name, color=colors.get(name, "black"), alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Ablation Study — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/ablation_curves.png", dpi=150)
    plt.close()
    print("Saved: plots/ablation_curves.png")


# =====================================================================
# Plot 3: Bar chart comparing mean episode rewards
# =====================================================================
def plot_mean_reward_comparison(eval_results):
    # eval_results is a dict: condition_name -> list of episode rewards
    names = list(eval_results.keys())
    means = [np.mean(eval_results[n]) for n in names]
    stds = [np.std(eval_results[n]) for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Mean Reward Comparison Across All Conditions")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/mean_reward_comparison.png", dpi=150)
    plt.close()
    print("Saved: plots/mean_reward_comparison.png")


# =====================================================================
# Plot 4: Sanction threshold sweep curves
# =====================================================================
def plot_sanction_curves():
    with open("models/sanction_rewards.pkl", "rb") as f:
        sanction_rewards = pickle.load(f)

    window = 20
    colors = {0.40: "tomato", 0.60: "steelblue", 0.80: "seagreen"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for threshold, rewards in sanction_rewards.items():
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))
        smoothed = np.array(smoothed)
        ax.plot(smoothed, label=f"threshold={threshold}", color=colors.get(threshold, "black"), alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Sanction Threshold Sweep — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/sanction_sweep_curves.png", dpi=150)
    plt.close()
    print("Saved: plots/sanction_sweep_curves.png")


# =====================================================================
# Plot 5: Map visualization using networkx
# =====================================================================
def plot_map_visualization():
    # I'll just draw the base graph with node labels and colors
    env = SovereignEnv()
    env.reset()

    G = env.G

    # positions — hand-coded to roughly match the topology in the strategy
    # this is approximate, just needs to look reasonable
    pos = {
        2: (0, 2),    # N0
        3: (-1, 1),   # C0
        8: (1, 1),    # C5
        5: (0, 0.5),  # C2
        0: (-1, -0.5),  # I0
        6: (1, -0.5),   # C3
        1: (2, -1),     # D0
        7: (-1, -1.5),  # C4
        4: (-1, -2.5),  # C1
    }

    # color by initial controller
    # I0 = invader (red), D0/C1/C3 = defender (blue), rest = neutral (gray)
    node_colors = []
    for node in G.nodes():
        ctrl = env._get_controller(node)
        if ctrl == 0:  # INVADER
            node_colors.append("tomato")
        elif ctrl == 1:  # DEFENDER
            node_colors.append("steelblue")
        else:
            node_colors.append("lightgray")

    labels = {}
    for node in G.nodes():
        labels[node] = NODE_NAMES[node]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(G, pos=pos, labels=labels, node_color=node_colors,
                     node_size=800, font_size=10, ax=ax,
                     edge_color="black", width=1.5)
    ax.set_title("SOVEREIGN Map — Initial State\n(Red=Invader, Blue=Defender, Gray=Neutral)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/map_visualization.png", dpi=150)
    plt.close()
    print("Saved: plots/map_visualization.png")


# =====================================================================
# Main evaluation loop
# =====================================================================
print("=" * 60)
print("Running evaluation")
print("=" * 60)

# evaluate all ablations
ablation_configs = [
    ("full",          {"use_legitimacy": True,  "use_occupation_cost": True,  "use_neutral_posture": True}),
    ("no_legitimacy", {"use_legitimacy": False, "use_occupation_cost": True,  "use_neutral_posture": True}),
    ("no_occ_cost",   {"use_legitimacy": True,  "use_occupation_cost": False, "use_neutral_posture": True}),
    ("no_neutral",    {"use_legitimacy": True,  "use_occupation_cost": True,  "use_neutral_posture": False}),
    ("baseline",      {"use_legitimacy": False, "use_occupation_cost": False, "use_neutral_posture": False}),
]

eval_results = {}  # will hold all eval rewards for bar chart

print("\n--- Evaluating ablations ---")
for name, kwargs in ablation_configs:
    model_path = f"models/ablation_{name}"
    if not os.path.exists(model_path + ".zip"):
        print(f"Skipping {name} — model not found at {model_path}.zip")
        continue

    rewards, pol_counts, mil_counts = evaluate_model(model_path, kwargs, n_episodes=100)
    eval_results[f"ablation_{name}"] = rewards

    # find dominant political and military actions
    dominant_pol = int(np.argmax(pol_counts))
    dominant_mil = int(np.argmax(mil_counts))

    print(f"\nAblation: {name}")
    print(f"  Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Dominant political action: {POL_ACTION_NAMES[dominant_pol]} ({pol_counts[dominant_pol]/pol_counts.sum()*100:.1f}%)")
    print(f"  Dominant military action: {MIL_ACTION_NAMES[dominant_mil]} ({mil_counts[dominant_mil]/mil_counts.sum()*100:.1f}%)")

# evaluate full model (seed 42)
print("\n--- Evaluating full model seeds ---")
for seed in [42, 123, 7]:
    model_path = f"models/full_model_seed{seed}"
    if not os.path.exists(model_path + ".zip"):
        print(f"Skipping seed {seed} — model not found")
        continue

    rewards, pol_counts, mil_counts = evaluate_model(
        model_path,
        {"use_legitimacy": True, "use_occupation_cost": True, "use_neutral_posture": True},
        n_episodes=100
    )
    eval_results[f"full_seed{seed}"] = rewards
    print(f"Full model seed {seed}: mean={np.mean(rewards):.3f} ± {np.std(rewards):.3f}")

# evaluate sanction threshold models
print("\n--- Evaluating sanction threshold models ---")
for threshold in [0.40, 0.60, 0.80]:
    threshold_str = str(threshold).replace(".", "_")
    model_path = f"models/sanction_{threshold_str}"
    if not os.path.exists(model_path + ".zip"):
        print(f"Skipping threshold {threshold} — model not found")
        continue

    rewards, pol_counts, mil_counts = evaluate_model(
        model_path,
        {"use_legitimacy": True, "use_occupation_cost": True, "use_neutral_posture": True, "sanction_threshold": threshold},
        n_episodes=100
    )
    eval_results[f"sanction_{threshold}"] = rewards
    print(f"Sanction threshold {threshold}: mean={np.mean(rewards):.3f} ± {np.std(rewards):.3f}")

# load and add baseline rewards to eval_results
print("\n--- Loading baseline results ---")
if os.path.exists("models/baseline_rewards.pkl"):
    with open("models/baseline_rewards.pkl", "rb") as f:
        baseline_rewards = pickle.load(f)
    for bl_name, bl_rewards in baseline_rewards.items():
        eval_results[f"baseline_{bl_name}"] = bl_rewards
        print(f"Baseline {bl_name}: mean={np.mean(bl_rewards):.3f} ± {np.std(bl_rewards):.3f}")

# =====================================================================
# Generate all plots
# =====================================================================
print("\n--- Generating plots ---")

if os.path.exists("models/full_model_rewards.pkl"):
    plot_full_model_curves()
else:
    print("Skipping full model curves — reward log not found")

if os.path.exists("models/ablation_rewards.pkl"):
    plot_ablation_curves()
else:
    print("Skipping ablation curves — reward log not found")

if os.path.exists("models/sanction_rewards.pkl"):
    plot_sanction_curves()
else:
    print("Skipping sanction curves — reward log not found")

if len(eval_results) > 0:
    plot_mean_reward_comparison(eval_results)
else:
    print("Skipping mean reward comparison — no eval results")

plot_map_visualization()

print("\nEvaluation complete. All plots saved to plots/")
