import torch
import numpy as np
from SevenPlus.Train.BaseTrainer import Trainer
from SevenPlus.Utils.distributions import action_output_to_distribution, calculate_log_action_probability


class PPO_Trainer(Trainer):
    def __init__(self, agents, buffer, net, decay_rate, action_space, state_space):
        super().__init__(agents, buffer)
        self.net = net
        self.decay_rate = decay_rate
        self.action_space = action_space
        self.state_space = state_space

        # parameters
        self.value_factor = 1.
        # self.value_factor = .5 # TODO: .5 in paper but 1 in openai baselines, .5 get better performance for some rough test. Can test more.
        self.entropy_factor = 0.01
        self.clip_epsilon = 0.2
        self.learning_rate = 3e-4
        self.training_epoch = 3
        # 512  # ENV_NUMBER x TIME_HORIZON should be lager than BATCH_SIZE
        self.time_horizon = 128
        self.batch_size = 256  # 4096  # can be bigger
        self.gamma = 0.99
        self.lam = 0.95

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate)

    def _apply_grad(self):
        pass

    def _compute_grad(self):
        pass

    def loss_fn(self):
        pass

    def set_dacay_rate(self, rate):
        self.decay_rate = rate

    def shuffle_data(self, states, actions, returns, values, log_probs, advs):
        indices = np.random.permutation(range(len(advs[0]))).tolist()
        states = states[:, indices]
        actions = actions[:, indices]
        returns = returns[:, indices]
        values = values[:, indices]
        log_probs = log_probs[:, indices]
        advs = advs[:, indices]
        return states, actions, returns, values, log_probs, advs

    def get_minibatch(self, index, sample_size, action_space, state_space, states, actions, returns, values, log_probs,
                      advs):
        batch_states = np.reshape(
            states[:, index:index + sample_size], [-1] + state_space)
        batch_actions = np.reshape(
            actions[:, index:index + sample_size], [-1] + action_space)
        batch_returns = np.reshape(
            returns[:, index:index + sample_size], [-1, 1])
        batch_values = np.reshape(
            values[:, index:index + sample_size], [-1, 1])
        batch_log_probs = np.reshape(
            log_probs[:, index:index + sample_size], [-1])
        batch_advs = np.reshape(advs[:, index:index + sample_size], [-1, 1])

        return batch_states, batch_actions, batch_returns, batch_values, batch_log_probs, batch_advs

    def _update(self):

        state_list, action_list, reward_list, terminal_list, value_list, log_prob_list = self.get_data()
        states, actions, rewards, terminals, values, log_probs = state_list, action_list, reward_list, terminal_list, value_list, log_prob_list
        advs = self.compute_GAE(rewards, values, terminals)
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
        returns = self.compute_reward_to_go_returns(rewards, values, terminals)

        for epoch in range(3):
            s, a, ret, v, logp, adv = self.shuffle_data(
                states, actions, returns, values, log_probs, advs)

            sample_size = 256
            for i in range(8 * 128 // sample_size):
                batch_s, batch_a, batch_ret, batch_v, batch_logp, batch_adv = \
                    self.get_minibatch(i * sample_size, sample_size, list(self.action_space.shape), list(self.state_space.shape), s,
                                       a, ret, v, logp, adv)
                state_batch = torch.from_numpy(batch_s)
                action_batch = torch.from_numpy(batch_a)
                return_batch = torch.from_numpy(batch_ret)
                old_value_batch = torch.from_numpy(batch_v)
                old_log_prob_batch = torch.from_numpy(batch_logp)
                adv_batch = torch.from_numpy(batch_adv)

                lr = self.learning_rate * self.decay_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # policy_head, value_batch = self.net(state_batch)
                logits_batch, value_batch = self.net(state_batch)
                # log_prob_batch = policy_head.log_prob(action_batch)
                action_distributions = action_output_to_distribution(
                    logits_batch)
                log_prob_batch = calculate_log_action_probability(
                    action_batch, action_distributions)
                self.v_loss, self.v_others = self.value_loss_clip(
                    value_batch, return_batch, old_value_batch)
                self.pi_loss, self.pi_others = self.policy_loss(
                    log_prob_batch, old_log_prob_batch, adv_batch)
                # self.entropy = torch.mean(policy_head.entropy())
                self.entropy = torch.mean(action_distributions.entropy())

                loss = self.v_loss * self.value_factor - \
                    self.pi_loss - self.entropy * self.entropy_factor

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

        return self.net

    def value_loss(self, value_batch, return_batch):
        value_loss = torch.mean((value_batch - return_batch) ** 2)
        others = None
        return value_loss, others

    def value_loss_clip(self, value_batch, return_batch, old_value_batch):
        value_clipped = old_value_batch + torch.clamp(value_batch - old_value_batch, -self.clip_epsilon,
                                                      self.clip_epsilon)
        value_loss_1 = (value_batch - return_batch) ** 2
        value_loss_2 = (return_batch - value_clipped) ** 2
        value_loss = .5 * torch.mean(torch.max(value_loss_1, value_loss_2))
        others = None
        return value_loss, others

    def policy_loss(self, log_prob_batch, old_log_prob_batch, adv_batch):
        ratio = torch.exp(log_prob_batch - old_log_prob_batch)
        ratio = ratio.view(-1, 1)  # take care the dimension here!!!
        surrogate_1 = ratio * adv_batch
        surrogate_2 = torch.clamp(ratio, 1 - self.clip_epsilon * self.decay_rate,
                                  1 + self.clip_epsilon * self.decay_rate) * adv_batch
        surrogate = torch.min(surrogate_1, surrogate_2)
        policy_loss = torch.mean(surrogate)

        approxkl = .5 * torch.mean((old_log_prob_batch - log_prob_batch) ** 2)
        clipfrac = torch.mean(
            torch.gt(torch.abs(ratio - 1.), self.clip_epsilon * self.decay_rate).float())
        others = {'approxkl': approxkl, 'clipfrac': clipfrac}
        return policy_loss, others

    def compute_reward_to_go_returns(self, rewards, values, terminals):
        '''
        the env will reset directly once it ends and return a new state
        st is only one more than at and rt at the end of the episode
        state:    s1 s2 s3 ... st-1 -
        action:   a1 a2 a3 ... at-1 -
        reward:   r1 r2 r3 ... rt-1 -
        terminal: t1 t2 t3 ... tt-1 -
        value:    v1 v2 v3 ... vt-1 vt
        '''
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

    def compute_GAE(self, rewards, values, terminals):
        # (N,T) -> (T,N)
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
        return advantages
