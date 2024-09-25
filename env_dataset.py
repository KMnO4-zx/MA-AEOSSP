import torch  # 导入 PyTorch 库，用于张量计算和深度学习
import numpy as np  # 导入 NumPy 库，用于数值计算和数组操作
from torch.utils.data import Dataset  # 从 PyTorch 导入 Dataset 类，用于数据集的定义
from gym.spaces import Discrete, Box, MultiDiscrete  # 从 Gym 导入离散、连续和多离散空间类
from ray.rllib.env import EnvContext  # 从 RLlib 导入环境上下文类，用于强化学习环境的配置
from ray import rllib  # 从 Ray 框架导入 RLlib 模块，用于强化学习算法和环境的处理
# 设置计算设备，如果有可用的 GPU 则使用 GPU，否则使用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SatelliteENV(rllib.MultiAgentEnv):  # 定义一个卫星环境类，继承自多智能体环境
    def __init__(self, config: EnvContext):  # 初始化环境，接受配置参数
        super(SatelliteENV, self).__init__()  # 调用父类的初始化方法
        self.batch_size = config['batch_size']  # 从配置中获取批处理大小
        self.instances_num = config['instances_num']  # 获取实例数量
        self.agent_num = config['agent_num']  # 获取智能体数量

        seed = np.random.randint(11111)  # 生成随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 随机种子
        self.transition_time = 0.8  # 设置任务转换时间（秒）
        self.transition_time = self.transition_time / 100.  # 转换时间归一化处理
        self.obs = None  # 初始化观察值
        self.priority = None  # 初始化任务优先级
        self.instances_state = torch.zeros((self.batch_size, 1, self.instances_num))  # 初始化实例状态为零

        self.max_steps = self.instances_num if self.update_mask is None else 1000  # 最大步数设置
        self.step_num = 0  # 步数计数器初始化

    def reset(self):  # 重置环境
        # 生成任务的持续时间
        duration = torch.rand(self.batch_size, self.instances_num + 1)  # 生成随机持续时间
        duration[:, 0] = 0.  # 设置第一个任务的持续时间为0
        # 生成任务的优先级
        shape = (self.batch_size, self.instances_num + 1)
        priority = torch.randint(1, 11, shape).float() / 10.  # 生成随机优先级（0.1到1.0之间）
        priority[:, 0] = 0.  # 第一个任务的优先级设为0
        self.priority = priority  # 保存优先级

        dynamic_1 = torch.zeros(shape[0], 2, shape[1])  # 初始化动态状态
        dynamic_1[:, 0, 0] = 2  # 设置第一个状态
        memory = torch.ones(shape[0], 1, shape[1])  # 初始化内存状态
        dynamic = torch.cat((dynamic_1, memory), dim=1)  # 合并动态和内存状态

        # 用于全局管理任务的状态
        self.instances_state = dynamic_1[:, 0, :]  # 更新实例状态

        obs = {}  # 初始化观察字典

        for agent_id in range(self.agent_num):  # 对每个智能体生成观察
            # TODO 后续把时间做成增量而不是随机
            # 生成时间窗
            x, y = self.generate_vtw(self.batch_size, self.instances_num)  # 生成时间窗
            static = torch.stack((x, y, duration, priority), 1)  # 合并静态信息
            obs[agent_id] = {"static": static, "dynamic": dynamic}  # 保存每个智能体的观察

        self.obs = obs  # 更新观察值
        self.step_num = 0  # 重置步数

        return obs  # 返回观察值
    
    def reward(self, chosen_ids):
        return 0
        # TODO 优化奖励函数, 加入冲突惩罚
        tour_idx = chosen_ids
        # tour_idx = torch.cat(act_ids, dim=1).cpu()  # (batch_size, node)
        # tour_idx_real_start = torch.cat(tour_idx_real_start, dim=1) * tour_idx.ne(0).float()  # (batch_size, node)
        # tour_idx_start = torch.cat(tour_idx_start, dim=1).float()   # (batch_size, node)
        # tour_idx_end = torch.cat(tour_idx_end, dim=1).float()   # (batch_size, node)

        # 卫星任务的收益(0.100-100-100-100.0)（每颗卫星共有m个任务）   数据 batch x node
        batch, node = self.priority.size()

        # 任务的收益百分比
        PRIreward = torch.zeros(batch)
        for i, act in enumerate(tour_idx):
            PRIreward[i] = self.priority[i, act].sum()

        sumPriority = self.priority.sum(1)
        reward_priority = 1 - PRIreward / sumPriority  # 收益百分比，0-1之间,越小越好
        return reward_priority
    
    def step(self, action_dict):  # 执行一步操作
        self.step_num += 1  # 增加步数
        obs, reward, done, info = {}, {}, False, {}  # 初始化输出变量

        if self.step_num == 1:  # 如果是第一步
            tour_before = False  # 未执行过巡回
        else:
            tour_before = True  # 已执行过巡回
        agent_actions = torch.zeros(self.batch_size, self.instances_num)  # 初始化智能体动作

        for agent_id, action in action_dict.items():  # 遍历每个智能体的动作
            # TODO 去除重复选择的ID
            agent_obs = self.obs[agent_id]  # 获取当前智能体的观察
            agent_obs["dynamic"][:, 0, :] = self.instances_state  # 更新动态状态
            agent_new_dynamic, task_start_time, task_end_time,\
                task_windows_start, task_windows_end \
                = self.update_dynamic(agent_obs["dynamic"], agent_obs["static"], action, tour_before)  # 更新动态状态

            self.instances_state = agent_new_dynamic[:, 0, :]  # 更新实例状态
            obs[agent_id] = {"static": agent_obs["static"], "dynamic": agent_new_dynamic}  # 更新观察

        self.obs = obs  # 更新环境观察
        reward = self.reward(agent_actions)  # 计算奖励

        mask = torch.ones(self.batch_size, self.instances_num + 1)  # 初始化掩码
        # 对mask进行约束，保证第一项一直置0
        state = self.instances_state[:]  # 获取当前状态
        state_id = state.nonzero()  # 找到非零状态的索引
        for i, j in state_id:  # 遍历所有非零状态
            mask[i, j] = 0  # 将掩码中对应的位置设为0
        mask[:, 0] = 0.  # 保证第一个任务的掩码位置为0
        mask = mask.to(device)  # 将掩码转移到计算设备上

        # 当到达最大步数或者任务都执行完成后结束
        if self.step_num == self.max_steps or not mask.byte().any():  # 检查是否结束
            done = True  # 标记为完成

        return obs, reward, done, {}  # 返回观察、奖励、完成标记和额外信息



if __name__ == "__main__":
    shape = (2, 4)
    dynamic_1 = torch.zeros(shape[0], 2, shape[1])
    dynamic_1[:, 0, 0] = 2
    memory = torch.ones(shape[0], 1, shape[1])
    dynamic = torch.cat((dynamic_1, memory), dim=1)
    print(dynamic.shape)
    print(dynamic_1.shape)
    print(dynamic_1)
    dynamic_1[:, 0, :] = 1
    print(dynamic_1[:, 0, :])
    print(dynamic_1)