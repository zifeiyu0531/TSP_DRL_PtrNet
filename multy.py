import numpy as np


class Multy:
    def __init__(self, data, cfg):
        self.data = data
        self.task_n = len(data)
        self.server_load = cfg.server_load
        self.alpha = cfg.alphaa
        self.beta = cfg.beta
        self.gama = cfg.gama

    def start(self):
        return self.get_multy_result()

    def get_multy_result(self):
        raw_tasks = []
        multy_run_map = [[] for _ in range(5)]
        time_slice = [5, 4, 3, 2, 2]
        for task in self.data:
            task = np.array(task)
            multy_run_map[int(task[4])].append(task)
        time_use = 0
        punish = 0
        ns_ = 0

        for i in range(5):
            server_ = [1, 1, 1, 1]
            temp_run_map = multy_run_map[i]
            if i == 4:  # 最后的队列
                time_use, ns_, raw_tasks = self.do_last_multy(temp_run_map, time_use, ns_, server_, raw_tasks, punish)
                break

            for task in temp_run_map:
                if time_use + task[-1] > task[-2]:
                    ns_ += 1
                    punish += task[-1] / self.server_load
                    raw_tasks.append(task)
                    continue
                if server_[0] < task[0] or server_[1] < task[1] \
                        or server_[2] < task[2] or server_[3] < task[3]:
                    time_use += time_slice[i]  # 一次并行完成，总时间增加
                    server_ = [1, 1, 1, 1]  # 归还资源

                server_ = np.subtract(server_, task[:4])  # 减去资源
                task[-1] -= time_slice[i]  # 减去时间片
                if task[-1] <= 0:  # 执行完毕
                    raw_tasks.append(task)
                else:
                    multy_run_map[i + 1].append(task)

        task_priority_max = 0
        for i in range(self.task_n):
            task_priority_max = max(task_priority_max, raw_tasks[i][4])

        task_priority_sum = 0
        for i in range(self.task_n):
            task_priority = raw_tasks[i][4]
            task_priority = (task_priority / task_priority_max) * (1 - i / self.task_n)
            task_priority_sum += task_priority

        ns_prob = ns_ / self.task_n
        reward = self.alpha * (time_use / (self.task_n * self.server_load)) + \
                 self.beta * (task_priority_sum / self.task_n) + \
                 self.gama * ns_prob
        return reward, time_use, task_priority_sum, ns_prob

    def do_last_multy(self, temp_run_map, time_use, ns_, server_remain, raw_tasks, punish):
        raw_tasks.extend(temp_run_map)
        server_run_map = []
        for task in temp_run_map:
            need = task[:4]
            time_out = task[5]
            time_need = task[6]

            if time_use + time_need > time_out:  # 超时
                ns_ += 1
                punish += time_need / self.server_load
                continue

            while server_remain[0] < need[0] or server_remain[1] < need[1] or \
                    server_remain[2] < need[2] or server_remain[3] < need[3]:
                server_run_map = np.array(server_run_map)
                time_use += 1  # 更新时间
                server_run_map[:, -1] -= 1
                server_run_map = server_run_map.tolist()

                while len(server_run_map) > 0:  # 移除已完成的任务
                    min_task_idx = np.argmin(server_run_map, axis=0)[-1]
                    min_task = server_run_map[min_task_idx]
                    min_need = min_task[:4]
                    min_time = min_task[-1]
                    if min_time > 0:
                        break
                    server_remain = np.add(server_remain, min_need)  # 更新剩余容量
                    del server_run_map[min_task_idx]  # 移除任务

            server_run_map.append(task)  # 将新任务加入服务器
            server_remain = np.subtract(server_remain, need)  # 更新服务器剩余容量

        max_time_idx = np.argmax(server_run_map, axis=0)[-1]
        max_time = server_run_map[max_time_idx][-1]
        time_use += max_time + punish
        return time_use, ns_, raw_tasks
