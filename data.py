import pymysql
import random
import numpy as np
from torch.utils.data.dataset import Dataset

db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='root', db='google_cluster_trace', charset='utf8')
cursor = db.cursor()


def get_instance(task_num=None):
    """
    获取单个任务队列
    :return: 获取到的任务队列
    """
    task_num = task_num if task_num else 500
    id_start = random.randint(0, 474000 - task_num - 1)
    cursor.execute(
        "SELECT * FROM google_cluster_trace.task_test WHERE id > %d LIMIT %d"
        % (id_start, task_num))
    task_list = cursor.fetchall()
    task_list = np.delete(task_list, 0, axis=1)
    task_list[:, 4] = abs(task_list[:, 4] - task_list[-1][4]) // 100000000
    return task_list


class Generator(Dataset):
    def __init__(self, cfg, env):
        self.data = env.get_batch_nodes(cfg.n_samples)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.size(0)
