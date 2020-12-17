import torch
import numpy as np
from SevenPlus.Train.BaseTrainer import Trainer

class Fake_Trainer(Trainer):
    def __init__(
            self,
            agents,
            buffer,
            net,
            decay_rate,
            action_space,
            state_space):
        super().__init__(agents, buffer)
        self.net = net

    def get_data(self):
        '''
        注意：当只有一个智能体，即当head_num=1时，返回数据维度为(M,T,...)
              当多余一个智能体，即当head_num>1时，返回数据维度为(N,M,T,...)
              其中N表示智能体数量，M表示环境数量，T为step数量

        即当head_num>1时的Returns
        Returns:
            states: 状态
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T,...)
            actions: 动作
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            rewards: 奖励
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            terminals: 终止
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            values: 值函数值
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            log_probs: log概率
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)

        即当head_num>1时的Returns
        Returns:
            states: 状态
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T,...)
            actions: 动作
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            rewards: 奖励
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            terminals: 终止
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            values: 值函数值
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            log_probs: log概率
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)

        '''

        head_num = TRAJ_HEAD_NUM if not ENABLE_MULTI_AGENT else MULTI_AGENTS_NUM
        print("**************", head_num)
        states = [getattr(self.buffer, "actor_{}_states".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_states.get_data()
        actions = [getattr(self.buffer, "actor_{}_actions".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_actions.get_data()
        rewards = [getattr(self.buffer, "actor_{}_rewards".format(i)).get_data(
        ) for i in range(head_num)] if head_num > 1 else self.buffer.actor_0_rewards.get_data()
        terminals = self.buffer.dones.get_data()
        values = [getattr(self.buffer, "actor_{}_values".format(i)).get_data(
        ) for i in range(head_num)] if head_num > 1 else self.buffer.actor_0_values.get_data()
        log_probs = [getattr(self.buffer, "actor_{}_logprobs".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_logprobs.get_data()
        print("********buffer数据**********")
        print("state_list: ", "shape:", np.array(states).shape)  # 3 x len(并行环境) x len(traj)
        print("action_list: ", "shape:", np.array(actions).shape)
        print("reward_list: ", "shape:", np.array(rewards).shape)
        print("terminal_list:", "shape:", np.array(terminals).shape)
        print("value_list: ", "shape:", np.array(values).shape)
        print("log_prob_list: ", "shape:", np.array(log_probs).shape)
        return states, actions, rewards, terminals, values, log_probs

    def _update(self):
        """梯度计算，更新网络参数"""
        self.get_data()
        return self.net

    def calculate_return(self):
        """return计算"""
        pass

    def data_to_tensor(self):
        """数据类型转换array->tensor， # N:环境数，T：step数"""
        pass

    def loss_fn(self):
        """定义损失函数"""
        pass

    def _apply_grad(self):
        pass

    def _compute_grad(self):
        pass

    def set_dacay_rate(self, rate):
        self.decay_rate = rate
