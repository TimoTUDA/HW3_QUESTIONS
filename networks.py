import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionValueCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """
        General action-value critic network architecture taking state and action as input.
        See https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py
        for reference.

        Args:
            input_shape (tuple).
            output_shape (tuple).
            n_features (int): number of features in a layer.
        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        # TODO: implement the network architecture with 2 hidden ReLU layers and one linear output layer
        # For ReLU layers, weights should be initialized xavier_uniform using nn.init.calculate_gain('relu')
        # For Linear layers, weights should be initialized xavier_uniform using nn.init.calculate_gain('linear')
        # [YOUR CODE]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)  # The output is Q-value which is scalar

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        # [YOUR CODE]
        state_action = torch.cat((state, action), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return q


class ValueCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """
        General value critic network architecture taking state as input.
        See https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py
        for reference.

        Args:
            input_shape (tuple).
            output_shape (tuple).
            n_features (int): number of features in a layer.
        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        # TODO: implement the network architecture with 2 hidden ReLU layers and one linear output layer
        # For ReLU layers, weights should be initialized xavier_uniform using nn.init.calculate_gain('relu')
        # For Linear layers, weights should be initialized xavier_uniform using nn.init.calculate_gain('linear')
        # [YOUR CODE]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)  # Output is a scalar value

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('linear'))




    def forward(self, state, **kwargs):
        # [YOUR CODE]
        features1 = F.relu(self._h1(state))
        features2 = F.relu(self._h2(features1))
        v = self._h3(features2)

        return v



class ActorNetwork(nn.Module):
    """
    General actor network architecture taking state as input.
    See https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py
    for reference.

    Args:
        input_shape (tuple).
        output_shape (tuple).
        n_features (int): number of features in a layer.
    """

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        # TODO: implement the network architecture with 2 hidden ReLU layers and one linear output layer
        # For ReLU layers, weights should be initialized xavier_uniform using nn.init.calculate_gain('relu')
        # For Linear layers, weights should be initialized xavier_uniform using nn.init.calculate_gain('linear')
        # [YOUR CODE]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('linear'))


    def forward(self, state):
        # TODO: implement forward pass of network
        # [YOUR CODE]
        features1 = F.relu(self._h1(state))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a
