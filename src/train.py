import os
import numpy as np
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from sovereign_env import SovereignEnv

# make sure models directory exists
os.makedirs("models", exist_ok=True)


# custom callback to track episode rewards during training
# I looked at the SB3 docs to figure out how to do this
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self):
        # accumulate reward each step
        # self.locals has the most recent reward from the env
        reward = self.locals["rewards"][0]
        self._current_episode_reward += reward

        # check if episode ended
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

        return True  # return True to keep training


def make_ppo(env, seed):
    # create a PPO model with the hyperparams from the strategy
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        verbose=1
    )
    return model


# =====================================================================
# Full model — 3 seeds
# =====================================================================
print("=" * 60)
print("Training full model with 3 seeds")
print("=" * 60)

full_model_rewards = {}

for seed in [42, 123, 7]:
    print(f"\n--- Full model, seed={seed} ---")
    env = SovereignEnv(use_legitimacy=True, use_occupation_cost=True, use_neutral_posture=True)
    model = make_ppo(env, seed)

    callback = RewardLoggerCallback()
    model.learn(total_timesteps=500_000, callback=callback)
    model.save(f"models/full_model_seed{seed}")

    # save the reward log too
    full_model_rewards[seed] = callback.episode_rewards
    print(f"Seed {seed} done. Episodes: {len(callback.episode_rewards)}")

# save reward logs for later plotting
with open("models/full_model_rewards.pkl", "wb") as f:
    pickle.dump(full_model_rewards, f)

print("\nFull model training done!")


# =====================================================================
# Ablation study — all 5 configs with seed=42
# =====================================================================
print("=" * 60)
print("Running ablation study")
print("=" * 60)

# format: (name, use_legitimacy, use_occupation_cost, use_neutral_posture)
ablation_configs = [
    ("full",          True,  True,  True),
    ("no_legitimacy", False, True,  True),
    ("no_occ_cost",   True,  False, True),
    ("no_neutral",    True,  True,  False),
    ("baseline",      False, False, False),
]

ablation_rewards = {}

for name, use_l, use_o, use_n in ablation_configs:
    print(f"\n--- Ablation: {name} ---")
    env = SovereignEnv(use_legitimacy=use_l, use_occupation_cost=use_o, use_neutral_posture=use_n)
    model = make_ppo(env, seed=42)

    callback = RewardLoggerCallback()
    model.learn(total_timesteps=500_000, callback=callback)
    model.save(f"models/ablation_{name}")

    ablation_rewards[name] = callback.episode_rewards
    print(f"Ablation {name} done. Episodes: {len(callback.episode_rewards)}")

with open("models/ablation_rewards.pkl", "wb") as f:
    pickle.dump(ablation_rewards, f)

print("\nAblation study done!")


# =====================================================================
# Sanction threshold sweep
# =====================================================================
print("=" * 60)
print("Running sanction threshold sweep")
print("=" * 60)

sanction_rewards = {}

for threshold in [0.40, 0.60, 0.80]:
    print(f"\n--- Sanction threshold: {threshold} ---")
    env = SovereignEnv(sanction_threshold=threshold)
    model = make_ppo(env, seed=42)

    callback = RewardLoggerCallback()
    model.learn(total_timesteps=500_000, callback=callback)
    # save with threshold in the name — need to handle the float for filename
    threshold_str = str(threshold).replace(".", "_")
    model.save(f"models/sanction_{threshold_str}")

    sanction_rewards[threshold] = callback.episode_rewards
    print(f"Threshold {threshold} done. Episodes: {len(callback.episode_rewards)}")

with open("models/sanction_rewards.pkl", "wb") as f:
    pickle.dump(sanction_rewards, f)

print("\nSanction sweep done!")


# =====================================================================
# Non-learning baselines
# =====================================================================
print("=" * 60)
print("Running non-learning baselines")
print("=" * 60)

# run each baseline for 100 episodes and collect rewards
NUM_BASELINE_EPISODES = 100

def run_baseline(policy_name, get_action_fn):
    env = SovereignEnv()
    all_rewards = []

    for ep in range(NUM_BASELINE_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = get_action_fn(env, obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward

        all_rewards.append(ep_reward)

    print(f"{policy_name}: mean={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.3f}")
    return all_rewards


# always invade: DO_NOTHING political, ADVANCE military
def always_invade_policy(env, obs):
    return [4, 0]  # DO_NOTHING=4, ADVANCE=0


# always negotiate: NEGOTIATE political, HOLD military
def always_negotiate_policy(env, obs):
    return [3, 1]  # NEGOTIATE=3, HOLD=1


# random policy
def random_policy(env, obs):
    return env.action_space.sample()


baseline_rewards = {}

baseline_rewards["always_invade"] = run_baseline("Always Invade", always_invade_policy)
baseline_rewards["always_negotiate"] = run_baseline("Always Negotiate", always_negotiate_policy)
baseline_rewards["random"] = run_baseline("Random", random_policy)

with open("models/baseline_rewards.pkl", "wb") as f:
    pickle.dump(baseline_rewards, f)

print("\nAll baselines done!")
print("\n========================================")
print("All training complete. Models saved to models/")
print("========================================")
