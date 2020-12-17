import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from gym.spaces import Box, Discrete

class MLPNetwork_init(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(MLPNetwork_init, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim

        self.hidden_dim = 64
        self.nonlin = F.relu

        self.norm_in = True
        self.discrete_action = True

        self.constrain_out_policy = True
        self.constrain_out_critic = False

        if self.norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(self.input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.out_dim)

        self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class MLP(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, action_space, observation_space):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLP, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.agent_init_params = self.get_dim()

        self.num_in_pol = self.agent_init_params[0]['num_in_pol']
        self.num_out_pol = self.agent_init_params[0]['num_out_pol']
        self.num_in_critic = self.agent_init_params[0]['num_in_critic']

        self.policy = MLPNetwork_init(self.num_in_pol, self.num_out_pol)

        self.critic = MLPNetwork_init(self.num_in_critic, 1)

        self.target_policy = MLPNetwork_init(self.num_in_pol, self.num_out_pol)

        self.target_critic = MLPNetwork_init(self.num_in_critic, 1)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=0.001)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=0.01)

    # TODO
    def observation_space_w(self):
        return [Box(low=0, high=50, shape=(136, 1)) for i in range(2)]

    def action_space_w(self):
        return [Discrete(5) for i in range(2)]

    def get_dim(self):
        agent_init_params = []
        self.action_space = self.action_space_w()
        self.observation_space = self.observation_space_w()
        for acsp, obsp in zip(self.action_space, self.observation_space):
            num_in_pol = obsp.shape[0]

            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n

            num_out_pol = get_shape(acsp)
            num_in_critic = 0

            for oobsp in self.observation_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in self.action_space:
                num_in_critic += get_shape(oacsp)

            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        return agent_init_params

    def forward(self, X, update=False):
        logits = self.policy(X)
        if update:
            value = self.critic(X)
            return logits, value
        else:
            return logits


# class MLPNetwork_init(nn.Module):
#     def __init__(self, input_dim, out_dim):
#
#         self.input_dim = input_dim
#         self.out_dim = out_dim
#
#         self.hidden_dim = 64
#         self.nonlin = F.relu
#
#         self.norm_in = True
#         self.discrete_action = True
#
#         self.constrain_out_policy = True
#         self.constrain_out_critic = False
#
#         if self.norm_in:  # normalize inputs
#             self.in_fn = nn.BatchNorm1d(input_dim)
#             self.in_fn.weight.data.fill_(1)
#             self.in_fn.bias.data.fill_(0)
#         else:
#             self.in_fn = lambda x: x
#
#         self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.fc3 = nn.Linear(self.hidden_dim, self.out_dim)
#
#         self.out_fn = lambda x: x
#
#     def forward(self, X):
#         """
#         Inputs:
#             X (PyTorch Matrix): Batch of observations
#         Outputs:
#             out (PyTorch Matrix): Output of network (actions, values, etc)
#         """
#         h1 = self.nonlin(self.fc1(self.in_fn(X)))
#         h2 = self.nonlin(self.fc2(h1))
#         out = self.out_fn(self.fc3(h2))
#         return out





