import numpy as np
import random


class Rand:
    def __init__(self, data):
        self.data = data

    def start(self):
        idx_list = list(range(len(self.data)))
        random.shuffle(idx_list)
        return idx_list


class Greed:
    def __init__(self, data):
        self.data = data

    def start(self):
        result_idx_list = []
        task_weight_list = []
        for task_ in self.data:
            task_weight_list.append([task_[0] + task_[1] + task_[2] + task_[3], task_[4], task_[5]])
        task_weight_list = np.array(task_weight_list)

        for i in range(len(self.data)):
            rand_idx = random.randint(0, 2)
            min_idx = -1
            if rand_idx == 0:
                min_idx = np.argmin(task_weight_list[:, 0])
            if rand_idx == 1:
                min_idx = np.argmin(task_weight_list[:, 1])
            if rand_idx == 2:
                min_idx = np.argmin(task_weight_list[:, 2])
            task_weight_list[min_idx, :] = 10000
            result_idx_list.append(min_idx)

        return result_idx_list
