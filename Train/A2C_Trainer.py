import torch
import numpy as np
from SevenPlus.Train.BaseTrainer import Trainer
from SevenPlus.Utils.distributions import action_output_to_distribution, calculate_log_action_probability

class A2C_Trainer(Trainer):

    def __init__(self, agents, buffer, net, decay_rate, action_space, state_space):
        """
        Args:
            #todo
            agents: agents列表
                data: [Actor(Agent_Ray,)]
                type: list
            buffer: 数据buffer
                data: <class 'SevenPlus.Train.Buffer.TrainerBuffer'>
                type: object
            net: 神经网络模型
                data: <class 'Examples.CartPole.Inference.MLP.MLP'>
                type: object
            decay_rate: 衰减学习率
                type: float
            action_space: 动作空间
                data: <class 'gym.spaces.discrete.Discrete'> Discrete/Box/...
                type: object
            state_space: 状态空间
                data: <class 'gym.spaces.box.Box'> Discrete/Box/...
                type: object
        """
        super().__init__(agents, buffer)
        self.net = net
        self.net.to(self.device)
        self.decay_rate = decay_rate
        self.action_space = action_space
        self.state_space = state_space

        # 定义算法参数
        self.value_factor = 0.5
        self.entropy_factor = 0.01
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.lam = 0.95

        # 定义优化器
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate)

    def calculate_return(self, rewards, values, terminals):
        """return计算
        根据reward计算return， # N:环境数，T：step数

        Args:
            rewards: 奖励
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)
            values: 值函数值
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)
            terminals: 终止
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)

        Returns:
            returns: 奖励
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)

        # (N,T) -> (T,N)   N:n_envs   T:traj_length
        the env will reset directly once it ends and return a new state
        st is only one more than at and rt at the end of the episode
        state:    s1 s2 s3 ... st-1 -
        action:   a1 a2 a3 ... at-1 -
        reward:   r1 r2 r3 ... rt-1 -
        terminal: t1 t2 t3 ... tt-1 -
        value:    v1 v2 v3 ... vt-1 vt
        """
        # (N,T) -> (T,N)   N:n_envs   T:traj_length
        rewards = np.transpose(rewards, [1, 0])
        values = np.transpose(values, [1, 0])
        terminals = np.transpose(terminals, [1, 0])
        R = values[-1]
        returns = []

        for i in reversed(range(rewards.shape[0])):
            R = rewards[i] + (1. - terminals[i]) * self.gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        # (T,N) -> (N,T)
        returns = np.transpose(returns, [1, 0])
        return returns

    def data_to_tensor(self, states, actions, returns, advantages):
        """数据类型转换array->tensor， # N:环境数，T：step数

        Args:
            states: 状态
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T,...)
            actions: 动作
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)
            returns: 奖励
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)
            advantages: 优势
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)

        Returns:
            state_tensor: 状态
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T,...]
            action_tensor: 动作
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T]
            return_tensor: 奖励
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T]
            advantage_tensor: 优势
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T]
        """
        # 加入trick, shuffle data
        # states, actions, returns, advantages, entropys = self.shuffle_data(states, actions, returns, advantages)
        indices = np.random.permutation(range(len(advantages[0]))).tolist()
        states = states[:, indices]
        actions = actions[:, indices]
        returns = returns[:, indices]
        advantages = advantages[:, indices]

        state_tensor = torch.from_numpy(states).to(self.device)
        action_tensor = torch.from_numpy(actions).to(self.device)
        return_tensor = torch.from_numpy(returns).to(self.device)
        advantage_tensor = torch.from_numpy(advantages).to(self.device)
        return state_tensor, action_tensor, return_tensor, advantage_tensor

    def loss_fn(self, state_tensor, action_tensor, return_tensor, advantage_tensor):
        """定义损失函数
        1.计算value损失
        2.计算策略损失
        3.计算熵损失
        4.计算总损失

        Args:
            state_tensor: 状态
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T,state_dim...]
            action_tensor: 动作
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T]
            return_tensor: 奖励
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T]
            advantage_tensor: 优势
                data: [[...],[...],...]
                type: torch.Tensor
                shape: [N,T]
        Returns:
            loss: 算法总损失
                type: torch.float64
        """

        # 1.计算value损失

        logits_batch, value_batch = self.net(state_tensor)
        self.v_loss = torch.mean(
            (value_batch.squeeze(-1) - return_tensor) ** 2)
        # 2.计算策略损失
        action_distributions = action_output_to_distribution(logits_batch)
        log_prob_batch = calculate_log_action_probability(
            action_tensor, action_distributions)
        self.pi_loss = (advantage_tensor * log_prob_batch).mean()
        # 3.计算熵损失
        self.entropy = torch.mean(action_distributions.entropy())
        # 4.计算总损失# 4.计算总损失
        loss = self.v_loss * self.value_factor - \
            self.pi_loss - self.entropy * self.entropy_factor
        # print(" self.v_loss:", self.v_loss.item(), "||", "self.pi_loss", self.pi_loss.item(), "||", "self.entropy",
        #       self.entropy.item())
        return loss

    def _update(self):
        """更新梯度
        1.获取loss
        2.optimizer.zero_grad() 清空过往梯度
        3.loss.backward() 反向传播，计算当前梯度
        4.nn.utils.clip_grad_norm(self.model.parameters) 梯度裁剪
        5.optimizer.step() 根据梯度更新网络参数

        Returns:
            net: 神经网络模型
                data: <class 'Examples.CartPole.Inference.MLP.MLP'>
                type: object
        """
        # 1. 获取loss
        state_list, action_list, reward_list, terminal_list, value_list, log_prob_list = self.get_data()
        advantages = self.compute_GAE(reward_list, value_list, terminal_list)
        returns = self.calculate_return(reward_list, value_list, terminal_list)
        state_tensor, action_tensor, return_tensor, advantages_tensor = self.data_to_tensor(
            state_list, action_list, returns, advantages)
        # 学习率衰减
        lr = self.learning_rate * self.decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        loss = self.loss_fn(state_tensor, action_tensor,
                            return_tensor, advantages_tensor)
        # 2.optimizer.zero_grad() 清空过往梯度
        self.optimizer.zero_grad()
        # 3.loss.backward() 反向传播，计算当前梯度
        loss.backward()
        # 4.nn.utils.clip_grad_norm(self.model.parameters) 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        # 5.optimizer.step() 根据梯度更新网络参数
        self.optimizer.step()
        return self.net

    """TRICKS"""

    def compute_GAE(self, rewards, values, terminals):
        """return计算
        根据reward计算return， # N:环境数，T：step数

        Args:
            rewards: 奖励
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)
            values: 值函数值
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)
            terminals: 终止
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)

        Returns:
            advantages: 优势
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (N,T)

        # (N,T) -> (T,N)   N:n_envs   T:traj_length
        the env will reset directly once it ends and return a new state
        st is only one more than at and rt at the end of the episode
        state:    s1 s2 s3 ... st-1 -
        action:   a1 a2 a3 ... at-1 -
        reward:   r1 r2 r3 ... rt-1 -
        terminal: t1 t2 t3 ... tt-1 -
        value:    v1 v2 v3 ... vt-1 vt
        """
        # (N,T) -> (T,N)   N:n_envs   T:traj_length
        rewards = np.transpose(rewards, [1, 0])
        values = np.transpose(values, [1, 0])
        terminals = np.transpose(terminals, [1, 0])
        length = rewards.shape[0]
        deltas = []
        for i in reversed(range(length)):
            v = rewards[i] + (1. - terminals[i]) * self.gamma * values[i + 1]
            delta = v - values[i]
            deltas.append(delta)
        deltas = np.array(list(reversed(deltas)))

        A = deltas[-1, :]
        advantages = [A]
        for i in reversed(range(length - 1)):
            A = deltas[i] + (1. - terminals[i]) * self.gamma * self.lam * A
            advantages.append(A)
        advantages = reversed(advantages)
        # (T,N) -> (N,T)
        advantages = np.transpose(list(advantages), [1, 0])
        advantages = (advantages - np.mean(advantages)) / \
            (np.std(advantages) + 1e-8)
        return advantages

    def set_dacay_rate(self, rate):
        self.decay_rate = rate

    def _apply_grad(self):
        pass

    def _compute_grad(self):
        pass