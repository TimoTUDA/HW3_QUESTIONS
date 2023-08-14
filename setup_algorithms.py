import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.algorithms.actor_critic import TD3, SAC, PPO

from mushroom_rl.policy import (
    OrnsteinUhlenbeckPolicy,
    GaussianTorchPolicy,
)
from networks import ActorNetwork, ActionValueCriticNetwork, ValueCriticNetwork


def setup_ppo_agent(mdp, n_features, lr, eps):
    """
    Set up the PPO agent with the specified parameters.
    You may use https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_trust_region.py as a reference.
    """
    # Actor
    actor_input_shape = mdp.info.observation_space.shape
    policy_params = dict(std_0=1.0, n_features=n_features, use_cuda=False)
    policy = GaussianTorchPolicy(
        ValueCriticNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        **policy_params
    )

    # Critic
    critic_input_shape = mdp.info.observation_space.shape
    critic_params = dict(
        network=ValueCriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr / 2}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Settings
    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr}}
    n_epochs_policy = 10
    batch_size = 64
    eps_ppo = eps
    lam = 0.99

    agent = PPO(
        mdp.info,
        policy,
        actor_optimizer,
        critic_params,
        n_epochs_policy,
        batch_size,
        eps_ppo,
        lam,
    )

    return agent


def setup_td3_agent(mdp, n_features, lr, policy_sigma):
    """
    Set up the TD3 agent with the specified parameters.
    You may use https://github.com/MushroomRL/mushroom-rl/blob/3fc832a27a93b5ba8e8b336b319616443e81794e/examples/pendulum_ddpg.py as a reference.
    """
    # Actor
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * policy_sigma, theta=0.15, dt=1e-2)

    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
    )
    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr / 2}}

    # Critic
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=ActionValueCriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Settings
    initial_replay_size = 2000
    max_replay_size = 200000
    batch_size = 64
    tau = 0.005

    # Agent
    agent = TD3(
        mdp.info,
        policy_class,
        policy_params,
        actor_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        tau,
    )
    return agent


def setup_sac_agent(mdp, n_features, lr, lr_alpha):
    """
    TODO:
    Set up the SAC agent with the specified parameters.
    You may use https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py as a reference.
    """

    # Actor
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
    )
    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr / 2}}

    # Critic
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=ActionValueCriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Settings
    initial_replay_size = 2000
    max_replay_size = 200000
    batch_size = 64
    tau = 0.005
    warmup_transitions = 1000

    # Agent
    agent = SAC(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
    )
    return agent
