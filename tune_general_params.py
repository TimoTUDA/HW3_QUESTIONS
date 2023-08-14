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
    setup_ppo_agent,
    setup_td3_agent,
    setup_sac_agent,
)
from torch import manual_seed


def single_experiment_general_params(log_dir, seed, n_features, learning_rate):
    # Seed
    np.random.seed(seed)
    manual_seed(seed)

    # MDP
    horizon = 250
    gamma = 0.99
    mdp = ThreeLinkedReacher(horizon, gamma)

    # Agent of your choice
    #THIS CAN BE CHANGED
    agent = setup_sac_agent(mdp, n_features=32, learning_rate=0.1)

    # Run single experiment
    Js, Vs, Hs = run_single_on_policy_experiment(log_dir, seed, agent, mdp)

    return Js, Vs, Hs


def main():
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_features", type=int)
    parser.add_argument("--learning_rate", type=float)

    args = parser.parse_args()
    results_dir = args.results_dir
    seed = int(args.seed)
    n_features = args.n_features
    learning_rate = args.learning_rate

    # Logging
    log_dir = Path(
        results_dir
        + f"/tune_general_params/n_features_{n_features}/learning_rate_{learning_rate}"
    )

    # Experiment
    Js, Vs, Hs = single_experiment_general_params(
        log_dir,
        seed,
        n_features,
        learning_rate,
    )

    # Save results
    np.save(Path(log_dir / f"Js-{seed}.npy"), Js)
    np.save(Path(log_dir / f"Vs-{seed}.npy"), Vs)
    np.save(Path(log_dir / f"Hs-{seed}.npy"), Hs)
    print(f"Saved: {log_dir}")


if __name__ == "__main__":
    main()
