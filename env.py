import torch
import numpy as np
import math
from tqdm import tqdm
import data
from param import *


def get_2city_distance(n1, n2):
    x1, y1, x2, y2 = n1[0], n1[1], n2[0], n2[1]
    if isinstance(n1, torch.Tensor):
        return torch.sqrt((x2 - x1).pow(2) + (y2 - y1).pow(2))
    elif isinstance(n1, (list, np.ndarray)):
        return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    else:
        raise TypeError


class Env:
    data = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self, cfg):
        '''
        nodes(cities) : contains nodes and their 2 dimensional coordinates
        [city_t, 2] = [3,2] dimension array e.g. [[0.5,0.7],[0.2,0.3],[0.4,0.1]]
        '''
        self.batch = cfg.batch
        self.task_n = cfg.task_n
        self.server_load = cfg.server_load
        self.alpha = cfg.alphaa
        self.beta = cfg.beta
        self.gama = cfg.gama

    def get_nodes(self, seed=None, task_n=100):
        '''
        return nodes:(task_n,2)
        '''
        if seed is not None:
            np.random.seed(seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Randomly generate (max_length) task
        multi_param = (15 * task_n) / self.server_load
        # [CPU, IO, Band, Memory]
        resource = np.random.rand(task_n, 4) * 2 / self.server_load
        task_priority = np.random.randint(5, size=(task_n, 1))
        time_use = np.random.randint(low=10, high=21, size=(task_n, 1))
        time_out = np.random.randint(low=round(multi_param * 0.8), high=round(multi_param * 1.2),
                                     size=(task_n, 1))
        samples = np.concatenate((resource, task_priority, time_out, time_use), axis=-1)
        samples = torch.tensor(samples, dtype=torch.float32, device=device)
        return samples

    def get_batch_nodes(self, n_samples):
        '''
        return nodes:(batch,task_n,6)
        '''
        if Env.data is not None:
            return Env.data
        # [n_samples, task_n, 6]
        samples = []
        part_samples = []
        print("start generate data")
        for _ in tqdm(range(n_samples // 10)):
            # [task_n, 6]
            instance = data.get_instance(self.task_n)
            part_samples.append(instance)
        for _ in range(10):
            samples.extend(part_samples)
        samples = torch.tensor(samples, dtype=torch.float32)
        print("generate data done")
        Env.data = samples
        return samples

    def stack_l_fast(self, inputs, tours):
        """
        *** this function is faster version of stack_l! ***
        inputs: (batch, task_n, 7), Coordinates of nodes
        tours: (batch, task_n), predicted tour
        d: (batch, task_n, 7)
        """
        inputs_cpu = inputs.cpu()
        # [batch_size, 6]
        result_list = []
        for task_list, idx_list in zip(inputs_cpu, tours):
            result = self.get_reward(task_list, idx_list)
            result_list.append(result)
        result_list = np.array(result_list)
        batch_reward = result_list[:, 0]
        return torch.tensor(batch_reward, dtype=torch.float32), np.mean(result_list, axis=0)

    def get_reward(self, task_list, idx_list):
        task_list = np.array(task_list)
        self.task_n = len(task_list)

        task_priority_max = 0
        for i in range(self.task_n):
            task_priority_max = max(task_priority_max, task_list[i][PRIORITY_IDX])
        task_priority_sum = 0
        for idx in range(self.task_n):
            i = idx_list[idx]
            task_priority = task_list[i][PRIORITY_IDX]
            task_priority = (task_priority / task_priority_max) * (1 - idx / self.task_n)
            task_priority_sum += task_priority

        cpu = 0
        time_use = 0
        waiting_time = 0
        server_run_map = []
        server_remain = np.array([1, 1, 1])
        for idx in idx_list:
            task = task_list[idx]
            need = task[:RESOURCE_NUM]

            while server_remain[0] < need[0] or server_remain[1] < need[1] or \
                    server_remain[2] < need[2]:
                server_run_map = np.array(server_run_map)
                time_use += 1  # 更新时间
                cpu += 1 - server_remain[0]
                server_run_map[:, -1] -= 1

                while len(server_run_map) > 0:  # 移除已完成的任务
                    min_task_idx = np.argmin(server_run_map, axis=0)[-1]
                    min_task = server_run_map[min_task_idx]
                    min_need = min_task[:RESOURCE_NUM]
                    min_time = min_task[-1]
                    if min_time > 0:
                        break
                    server_remain = np.add(server_remain, min_need)  # 更新剩余容量
                    server_run_map = np.delete(server_run_map, min_task_idx, axis=0)  # 移除任务

            # 资源充足，直接下放任务
            if len(server_run_map) == 0:
                server_run_map = np.array([task])
            else:
                server_run_map = np.row_stack((server_run_map, task))
            waiting_time += task[RELEASE_TIME_IDX] + time_use
            server_remain = np.subtract(server_remain, need)  # 更新服务器剩余容量

        # 运行完剩余任务
        while len(server_run_map) > 0:
            cpu = np.sum(server_run_map, axis=0)[0]
            server_run_map = np.array(server_run_map)
            time_use += 1
            server_run_map[:, TIME_IDX] -= 1
            # 移除已执行完的任务
            while len(server_run_map) > 0 and np.min(server_run_map, axis=0)[TIME_IDX] == 0:
                min_task_idx = np.argmin(server_run_map, axis=0)[TIME_IDX]
                server_run_map = np.delete(server_run_map, min_task_idx, axis=0)

        cpu = cpu / time_use
        waiting_time_avg = waiting_time / self.task_n
        reward = cpu + time_use / self.task_n + task_priority_sum / self.task_n + waiting_time_avg / 50
        return [reward, cpu, time_use, task_priority_sum, waiting_time_avg]
