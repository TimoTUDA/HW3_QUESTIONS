import os
import argparse
import numpy as np
from pathlib import Path
from mushroom_rl_extensions.environments.mujoco_envs.three_linked_reacher import (
    ThreeLinkedReacher,
)
from run_single_experiment import (
    run_single_on_policy_experiment,
    run_single_off_policy_experiment,
)
from setup_algorithms import (
    setup_td3_agent,
    setup_sac_agent,
    setup_ppo_agent,
)
from torch import manual_seed


def tune_ppo(log_dir, seed, n_features, learning_rate, eps):
    # Seed
    np.random.seed(seed)
    manual_seed(seed)

    # MDP
    horizon = 250
    gamma = 0.99
    mdp = ThreeLinkedReacher(horizon, gamma)

    # Agent
    agent = setup_ppo_agent(mdp, n_features=n_features, lr=learning_rate, eps=eps)

    # Run single experiment
    Js, Vs, Hs = run_single_on_policy_experiment(log_dir, seed, agent, mdp)

    return Js, Vs, Hs


def tune_td3(log_dir, seed, n_features, learning_rate, policy_sigma):
    # Seed
    np.random.seed(seed)
    manual_seed(seed)

    # MDP
    horizon = 250
    gamma = 0.99
    mdp = ThreeLinkedReacher(horizon, gamma)

    # Agent
    agent = setup_td3_agent(
        mdp, n_features=n_features, lr=learning_rate, policy_sigma=policy_sigma
    )

    # Run single experiment
    Js, Vs, Hs = run_single_off_policy_experiment(log_dir, seed, agent, mdp)

    return Js, Vs, Hs


def tune_sac(log_dir, seed, n_features, learning_rate, lr_alpha):
    # Seed
    np.random.seed(seed)
    manual_seed(seed)

    # MDP
    horizon = 250
    gamma = 0.99
    mdp = ThreeLinkedReacher(horizon, gamma)

    # Agent
    agent = setup_sac_agent(
        mdp, n_features=n_features, lr=learning_rate, lr_alpha=lr_alpha
    )

    # Run single experiment
    Js, Vs, Hs = run_single_off_policy_experiment(log_dir, seed, agent, mdp)

    return Js, Vs, Hs


def main():
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--algorithm", type=str, choices=["ppo", "td3", "sac"])
    parser.add_argument("--hyperparam_value", type=float)

    args = parser.parse_args()
    results_dir = args.results_dir
    seed = args.seed
    algorithm = args.algorithm
    hyperparam_value = args.hyperparam_value

    # Logging
    log_dir = Path(results_dir + f"/tune_{algorithm}/{hyperparam_value}")

    # Experiment
    n_features = # YOUR CHOSEN VALUE
    learning_rate = # YOUR CHOSEN VALUE
    if algorithm == "ppo":
        Js, Vs, Hs = tune_ppo(
            log_dir, seed, n_features, learning_rate, hyperparam_value
        )
        np.save(Path(log_dir / f"Js-{seed}.npy"), Js)
        np.save(Path(log_dir / f"Vs-{seed}.npy"), Vs)
        np.save(Path(log_dir / f"Hs-{seed}.npy"), Hs)
        print(f"Saved: {log_dir}")
    elif algorithm == "td3":
        Js, Vs, Hs = tune_td3(log_dir, seed, n_features, learning_rate, hyperparam_value)
        np.save(Path(log_dir / f"Js-{seed}.npy"), Js)
        np.save(Path(log_dir / f"Vs-{seed}.npy"), Vs)
        np.save(Path(log_dir / f"Hs-{seed}.npy"), Hs)
        print(f"Saved: {log_dir}")
    elif algorithm == "sac":
        Js, Vs, Hs = tune_sac(
            log_dir, seed, n_features, learning_rate, hyperparam_value
        )
        np.save(Path(log_dir / f"Js-{seed}.npy"), Js)
        np.save(Path(log_dir / f"Vs-{seed}.npy"), Vs)
        np.save(Path(log_dir / f"Hs-{seed}.npy"), Hs)
        print(f"Saved: {log_dir}")
    else:
        raise ValueError("Invalid algorithm provided!")


if __name__ == "__main__":
    main()
