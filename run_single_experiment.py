import numpy as np
from tqdm import trange

from mushroom_rl.core import Core
from mushroom_rl.core.logger import Logger
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl_extensions.utils.dataset import compute_Q, compute_V, compute_H


def run_single_off_policy_experiment(
    log_dir,
    seed,
    agent,
    mdp,
    n_epochs=100,
    n_steps_training_per_epoch=10000,
    n_episodes_test_per_epoch=10,
):
    # Logger
    logger = Logger(
        log_name="logger",
        results_dir=log_dir,
        log_console=True,
        seed=seed,
        console_log_level=30,
    )
    logger.strong_line()
    logger.info("Experiment Algorithm: " + type(agent).__name__)

    # Algorithm
    core = Core(agent, mdp)

    # Metrics
    Js = []
    Rs = []
    Vs = []
    Hs = []

    # TODO: Evaluate agent performance before training
    # Evaluation before training
    dataset = dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch)# [YOUR EVALUATION CODE]
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    Js.append(J)
    R = np.mean(compute_J(dataset))
    Rs.append(R)
    V = np.mean(compute_Q(dataset, agent))
    Vs.append(V)
    H = np.mean(compute_H(dataset, agent))
    Hs.append(H)
    logger.epoch_info(0, J=J, R=R, V=V, H=H)

    core.learn(n_steps=2000, n_steps_per_fit=2000)  # Off-policy warmup transitions
    for n in trange(n_epochs, leave=False):
        # TODO: use the learn and evaluate functions in the MushroomRL core object
        # to train the agent for n_steps_training_per_epoch steps and get a sample
        # of n_episodes_test_per_epoch test episodes in a dataset

        # [YOUR TRAINING CODE]
        core.learn(n_steps=n_steps_training_per_epoch)

        dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch) # [YOUR EVALUATION CODE]
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        Js.append(J)
        R = np.mean(compute_J(dataset))
        Rs.append(R)
        V = np.mean(compute_Q(dataset, agent))
        Vs.append(V)
        H = np.mean(compute_H(dataset, agent))
        Hs.append(H)

        # Log info & save best agent
        logger.epoch_info(n + 1, J=J, R=R, V=V, H=H)
        logger.log_best_agent(agent, J)
    return Js, Vs, Hs


def run_single_on_policy_experiment(
    log_dir,
    seed,
    agent,
    mdp,
    n_epochs=100,
    n_steps_training_per_epoch=10000,
    n_episodes_test_per_epoch=10,
):
    # Logger
    logger = Logger(
        log_name="logger",
        results_dir=log_dir,
        log_console=True,
        seed=seed,
        console_log_level=30,
    )
    logger.strong_line()
    logger.info("Experiment Algorithm: " + type(agent).__name__)

    # Algorithm
    core = Core(agent, mdp)

    # Metrics
    Js = []
    Rs = []
    Vs = []
    Hs = []

    # TODO: Evaluate agent performance before training
    # Evaluation before training
    dataset = dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch)# [YOUR EVALUATION CODE]
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    Js.append(J)
    R = np.mean(compute_J(dataset))
    Rs.append(R)
    V = np.mean(compute_V(dataset, agent._V))
    Vs.append(V)
    H = np.mean(compute_H(dataset, agent))
    Hs.append(H)
    logger.epoch_info(0, J=J, R=R, V=V, H=H)

    for n in trange(n_epochs, leave=False):
        # TODO: use the learn and evaluate functions in the MushroomRL core object
        # to train the agent for n_steps_training_per_epoch steps and get a sample
        # of n_episodes_test_per_epoch test episodes in a dataset

        core.learn(n_steps=n_steps_training_per_epoch)# [YOUR TRAINING CODE]

        dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch)# [YOUR EVALUATION CODE]
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        Js.append(J)
        R = np.mean(compute_J(dataset))
        Rs.append(R)
        V = np.mean(compute_V(dataset, agent._V))
        Vs.append(V)
        H = np.mean(compute_H(dataset, agent))
        Hs.append(H)

        # Log info & save best agent
        logger.epoch_info(n + 1, J=J, R=R, V=V, H=H)
        logger.log_best_agent(agent, J)
    return Js, Vs, Hs
