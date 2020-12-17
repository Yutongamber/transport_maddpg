from SevenPlus.Wraper.BaseWraper import BaseWrapper
import gym
import numpy as np
# from transport import Transport
from Examples.transport_7plus.Env.transport import Transport
from SevenPlus.Launcher.LauncherConfig import config as g_config

# TODO conf
name = g_config.get_custom('transport.name')
conf = g_config.get_custom('transport.conf')
conf = {
    'n_player': 2,#玩家数量
    'board_width': 11,#地图宽
    'board_height': 11,#地图高
    'n_cell_type': 5,#格子的种类
    'materials': 4,#集散点数量
    'cars': 2,#汽车数
    'planes': 0,#飞机数量
    'barriers': 12,#固定障碍物数量
    'max_step' :100,#最大步数
    'game_name':name,#游戏名字
    'K': 5,#每个K局更新集散点物资数目
    'map_path': 'Examples/transport_7plus/Env/map.txt',#存放初始地图
    'cell_range': 6,  # 单格中各维度取值范围（tuple类型，只有一个int自动转为tuple）##?
    'ob_board_width': None,  # 不同智能体观察到的网格宽度（tuple类型），None表示与实际网格相同##?
    'ob_board_height': None,  # 不同智能体观察到的网格高度（tuple类型），None表示与实际网格相同##?
    'ob_cell_range': None,  # 不同智能体观察到的单格中各维度取值范围（二维tuple类型），None表示与实际网格相同##?
}


class TransportEnvv(BaseWrapper):
    def __init__(self, character, train_side=0):  # TODO: to check
        self.env = Transport(conf)
        super().__init__(self.env)
        self.character = character
        self.train_side = self.character if self.character != -1 else train_side
        if self.train_side == -1:
            raise AttributeError("train_side set error")

    def get_observationspace(self, character=None):
        if character != -1:
            return self.env.observation_space
        return self.env.observation_space

    def get_actionspace(self, character=None):
        if character != -1:
            return self.env.action_space
        return self.env.action_space

    def reset(self):
        state = self.env.reset()
        Obss = []
        for n in range(conf['cars']):
            Obss.append(
                np.array(state[0:133] + state[(133 + 3 * n):(136 + 3 * n)]))
        return Obss

    def modify_action(self):
        pass

    def step(self, action):
        """

        Args:
            joint_action:

        Returns:
            Obss: [ array([]),..., array([])]
            Reward: [ r1, r2,..., rn]

        """

        # action_wrapper:
        joint_action = []

        ## add: 前两个action
        for i in range(2):
            player = []
            for j in range(1):
                each = [0] * 11
                # idx = np.random.randint(11)
                each[3] = 1
                player.append(each)
            joint_action.append(player)
        # print('################### debug action: ', action)
        for m in range(conf['cars']):
            each = [0] * 5
            # print("action list is {}".format(action[m].astype(int).tolist()))
            one_hot_list = action[m].astype(int).tolist()
            #########
            # index = 0
            # try:
            #     index = one_hot_list[0].index(1)
            # except:
            #     pass
            #########
            # print('############### debug one hot: ', one_hot_list)
            index = one_hot_list[0].index(1)
            each[index] = 1
            agent_action = [each]
            joint_action.append(agent_action)
        # print('joint_action', joint_action)

        state, reward, done, info = self.env.step(joint_action)

        # state_wrapper
        Obss = []
        for n in range(conf['cars']):
            Obss.append(
                np.array(state[0:133] + state[(133 + 3 * n):(136 + 3 * n)]))

        # reward_wrapper
        Reward = reward
        # Reward = []
        # for n in range(conf['cars']):
        #     Reward.append(reward[n])

        change_info = {}
        change_info["result"] = info
        return Obss, Reward, done, change_info

    def seed(self):  # TODO: to add
        pass

    def render(self):
        pass

    def close(self):
        pass

    def over(self):
        return self.env.is_ternimal()

