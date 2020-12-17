import torch
import torch.nn.functional as F
import numpy as np
from SevenPlus.Train.BaseTrainer import Trainer
from torch.autograd import Variable
# from ddpg import DDPGAgent

from SevenPlus.Launcher.LauncherConfig import config as g_config
import torchsnooper

MSELoss = torch.nn.MSELoss()
ENABLE_MULTI_AGENT = g_config.get_setting('reinforcement.multi_agents.enable')
MULTI_AGENTS_NUM = g_config.get_setting(
    'reinforcement.multi_agents.agents_num')
TRAJ_HEAD_NUM = g_config.get_setting(
    'reinforcement.agent.rollout.traj_head_num')


class MADDPGTrainer(Trainer):
    def __init__(self, agents, buffer, net, decay_rate, action_space, state_space):
        super().__init__(agents, buffer)

        self.net = net #一个list
        self.gamma = 0.98
        self.tau = 0.01
        self.discrete_action = True
        self.hidden_dim = 64

        self.agent_init_params = [{'num_in_pol': 128, 'num_out_pol': 5, 'num_in_critic': 266},
        {'num_in_pol': 128, 'num_out_pol': 5, 'num_in_critic': 266}]

        self.nagents = 2  # TODO
        self.alg_types = 'MADDPG'
        self.filled_i = 2000
        self.mini_batch = 1024

        # 设备设定
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    def get_data(self):
        '''
        注意：当只有一个智能体，即当head_num=1时，返回数据维度为(M,T,...)
              当多余一个智能体，即当head_num>1时，返回数据维度为(N,M,T,...)
              其中N表示智能体数量，M表示环境数量，T为step数量

        即当head_num=1时的Returns
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
        states = [getattr(self.buffer, "actor_{}_states".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_states.get_data()
        # print("buffer in trainer is {}".format(self.buffer.get_data()))
        actions = [getattr(self.buffer, "actor_{}_actions".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_actions.get_data()
        rewards = [getattr(self.buffer, "actor_{}_rewards".format(i)).get_data(
        ) for i in range(head_num)] if head_num > 1 else self.buffer.actor_0_rewards.get_data()
        terminals = self.buffer.dones.get_data()
        values = [getattr(self.buffer, "actor_{}_values".format(i)).get_data(
        ) for i in range(head_num)] if head_num > 1 else self.buffer.actor_0_values.get_data()
        log_probs = [getattr(self.buffer, "actor_{}_logprobs".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_logprobs.get_data()
        # print("********buffer数据**********")
        # print("state_list: ", "shape:", np.array(states).shape)  # 3 x len(并行环境) x len(traj)
        # print("action_list: ", "shape:", np.array(actions).shape)
        # print("reward_list: ", "shape:", np.array(rewards).shape)
        # print("terminal_list:", "shape:", np.array(terminals).shape)
        # print("value_list: ", "shape:", np.array(values).shape)
        # print("log_prob_list: ", "shape:", np.array(log_probs).shape)
        # print("get data states is {}".format(np.array(states[:][:][:-2][:]).shape))
        # print("get data states is {}".format(np.array(states[:][:][:][:-1]).shape))
        # print(np.array(states[:-1][:][:][:]).shape)
        # print(np.array(states[:][:-1][:][:]).shape)

        states_np = np.array(states)[:, :, :-1, :].tolist()
        next_np = np.array(states)[:, :, 1:, :].tolist()

        # print("********buffer out 数据**********")
        # print("state_list: ", "shape:", np.array(states_np).shape)  # 3 x len(并行环境) x len(traj)
        # print("action_list: ", "shape:", np.array(actions).shape)
        # print("reward_list: ", "shape:", np.array(rewards).shape)
        # print("terminal_list:", "shape:", np.array(terminals).shape)
        # print("value_list: ", "shape:", np.array(next_np).shape)
        # print("log_prob_list: ", "shape:", np.array(log_probs).shape)

        return states_np, actions, rewards, next_np, terminals, log_probs
        # return states[:, :, 0:-1, :], actions, rewards, terminals, states[:, :, 1:124, :], log_pro

    def sample(self, data, N, to_gpu=False, norm_rews=True):
        # print("data length in trainer is {}".format(len(data)))
        # print("data tensor shape is {}".format(torch.Tensor(data[0]).size()))
        #
        # print("In sample p obs size is {}".format(torch.Tensor(data[0]).size()))
        # print("In sample p acs size is {}".format(torch.Tensor(data[1]).size()))
        # print("In sample p rew size is {}".format(torch.Tensor(data[2]).size()))
        # print("In sample p next_obs is {}".format(torch.Tensor(data[3]).size()))
        # print("In sample p dones is {}".format(torch.Tensor(data[4]).size()))

        obs = torch.Tensor(data[0]).permute(1, 0, 2, 3) # obs
        action = torch.Tensor(data[1]).squeeze(3).permute(1, 0, 2, 3) # action
        reward = torch.Tensor(data[2]).permute(1, 0, 2, 3) # reward
        obs_ = torch.Tensor(data[3]).permute(1, 0, 2, 3) # next_obs
        done = torch.Tensor(data[4]).permute(1, 0)  # done # ???

        # 无重复采样N
        inds = np.random.choice(self.filled_i, size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(x, requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(x, requires_grad=False)

        if norm_rews:
            # print("data[2] size is {}".format(data[2].size()))
            # print("data[2] size is {}".format(data[2][0][0].size()))
            # print("data[2] size is {}".format(data[2][:,:,50,1]))
            ret_rews = [cast((reward[:,i,inds,i] -
                              reward[:,i,:self.filled_i,:].mean()) /
                             (reward[:,i,:self.filled_i,:].std() + 0.1))

                        for i in range(self.nagents)]
        else:
            ret_rews = [cast(reward[i][inds]) for i in range(self.nagents)]
        
        return ([cast(obs[:,i,inds,:]) for i in range(self.nagents)],
                [cast(action[:,i,inds,:]) for i in range(self.nagents)],
                ret_rews,
                [cast(obs_[:,i,inds,:]) for i in range(self.nagents)],
                [cast(done[inds, :]) for i in range(self.nagents)])

    def get_data_agent(self):
        obs, acs, rews, next_obs, dones, _= self.get_data()
        # print(obs.shape, acs.shape, rews.shape, next_obs.shape, dones.shape)
        data = self.sample([obs, acs, rews, next_obs, dones], self.mini_batch)
        return data
    #不变
    def step(self):
        self._checkBuffer()
        # self.data_length = len(self.buffer.rewards_0.get_data()[1])
        tmp = self.buffer.dones.get_storage()
        if len(tmp) > 0:
            self.data_length = len(tmp[0])
        if self.data_length >= 1:
            model = self._update()
            self.data_length = 0
            # self.set_policy(model)

            return model
        else:
            # time.sleep(3)
            return None


    def _update(self):

        # TODO get_data to data # 可以buffer打乱 #也可
        # TODO 加if限制
        data = self.get_data_agent()
        for a_i in range(len(self.net)):
            self.update(a_i, data, logger=None)
        self.update_all_targets()
        return self.net

    @property
    def target_policies(self):
        return [a.target_policy for a in self.net]
    @property
    def policies(self):
        return [a.policy for a in self.net]

    def update(self, agent_i, data, logger=None):
        #print('*************************************************************************************')
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        #TODO
        obs, acs, rews, next_obs, dones = data
        # print('---------------debug dones: ', dones)
        curr_agent = self.net[agent_i]
        
        curr_agent.critic_optimizer.zero_grad()

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [self.onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)]
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                         next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=2)

        # view: 得到的是一个列tensor
        target_value = (rews[agent_i][:,:].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in).view(-1, 1) *
                        (1 - dones[agent_i].view(-1, 1)))
        #print('---------------debug target_value: ', target_value)
        vf_in = torch.cat((*obs, *acs), dim=2)

        actual_value = curr_agent.critic(vf_in)
        #print('---------------debug actual_value: ', actual_value.view(-1, 1))
        vf_loss = MSELoss(actual_value.view(-1, 1), target_value.detach())
        #print('---------------debug vf_loss: ', vf_loss)
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        #print('############################# policy ###################################')
        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = self.gumbel_softmax(curr_pol_out, hard=True) # softmax + argmax
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(self.onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob))
        vf_in = torch.cat((*obs, *all_pol_acs), dim=2)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        #print('---------------debug pol_loss: ', pol_loss)
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.net:
            self.soft_update(a.target_critic, a.critic, self.tau)
            self.soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

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

    def onehot_from_logits(self, logits, eps=0.0):
        """
        Given batch of logits, return one-hot sample using epsilon greedy strategy
        (based on given epsilon)
        """
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                            enumerate(torch.rand(logits.shape[0]))])

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = self.onehot_from_logits(y)
            y = (y_hard - y).detach() + y
        return y

    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.shape, tens_type=type(logits.data))
        return F.softmax(y / temperature, dim=1)

    def sample_gumbel(self, shape, eps=1e-20, tens_type=torch.FloatTensor):
        """Sample from Gumbel(0, 1)"""
        U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
        return -torch.log(-torch.log(U + eps) + eps)