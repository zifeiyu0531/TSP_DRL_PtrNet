from env import Env
from config import Config, load_pkl, pkl_parser
from search import sampling
from rand_greed import Rand, Greed
from multy import Multy
import matplotlib.pyplot as plt
from tqdm import tqdm


def start_scheduling(input, cfg, env):
    pred_tour = sampling(cfg, env, input)
    test_input = input.cpu()
    reward_, time_use_, priority_sum_, ns_prob_ = env.get_reward(test_input, pred_tour)
    return reward_, time_use_, priority_sum_, ns_prob_


def start_rand(input, env):
    idx_list = Rand(input).start()
    reward_, time_use_, priority_sum_, ns_prob_ = env.get_reward(test_input, idx_list)
    return reward_, time_use_, priority_sum_, ns_prob_


def start_greed(input, env):
    idx_list = Greed(input).start()
    reward_, time_use_, priority_sum_, ns_prob_ = env.get_reward(test_input, idx_list)
    return reward_, time_use_, priority_sum_, ns_prob_


def start_multy(input, cfg):
    reward_, time_use_, priority_sum_, ns_prob_ = Multy(input, cfg).start()
    return reward_, time_use_, priority_sum_, ns_prob_


if __name__ == '__main__':
    cfg = load_pkl(pkl_parser().path)
    env = Env(cfg)

    if cfg.mode == 'test':
        # 服务器平均负载服务请求个数：60~100
        reward_list, time_use_list, priority_sum_list, ns_prob_list = [], [], [], []
        reward_rand_list, time_use_rand_list, priority_sum_rand_list, ns_prob_rand_list = [], [], [], []
        reward_greed_list, time_use_greed_list, priority_sum_greed_list, ns_prob_greed_list = [], [], [], []
        reward_multy_list, time_use_multy_list, priority_sum_multy_list, ns_prob_multy_list = [], [], [], []

        for task_n in tqdm(range(60, 101)):
            for _ in range(5):
                test_input = env.get_nodes(task_n=task_n)

                reward, time_use, priority_sum, ns_prob = start_scheduling(test_input, cfg, env)
                reward_list.append(reward)
                time_use_list.append(time_use)
                priority_sum_list.append(priority_sum)
                ns_prob_list.append(ns_prob)
                test_input = test_input.cpu()

                reward_rand, time_use_rand, priority_sum_rand, ns_prob_rand = start_rand(test_input, env)
                reward_rand_list.append(reward_rand)
                time_use_rand_list.append(time_use_rand)
                priority_sum_rand_list.append(priority_sum_rand)
                ns_prob_rand_list.append(ns_prob_rand)

                reward_greed, time_use_greed, priority_sum_greed, ns_prob_greed = start_greed(test_input, env)
                reward_greed_list.append(reward_greed)
                time_use_greed_list.append(time_use_greed)
                priority_sum_greed_list.append(priority_sum_greed)
                ns_prob_greed_list.append(ns_prob_greed)

                reward_multy, time_use_multy, priority_sum_multy, ns_prob_multy = start_multy(test_input, cfg)
                reward_multy_list.append(reward_multy)
                time_use_multy_list.append(time_use_multy)
                priority_sum_multy_list.append(priority_sum_multy)
                ns_prob_multy_list.append(ns_prob_multy)

        #         for task_n in tqdm([100, 83, 71, 62, 56]):
        #             test_input = env.get_nodes(cfg.seed, task_n)

        #             reward, time_use, priority_sum, ns_prob = start_scheduling(test_input, cfg, env)
        #             reward_list.append(reward)
        #             time_use_list.append(time_use)
        #             priority_sum_list.append(priority_sum)
        #             ns_prob_list.append(ns_prob)
        #             test_input = test_input.cpu()

        #             reward_rand, time_use_rand, priority_sum_rand, ns_prob_rand = start_rand(test_input, env)
        #             reward_rand_list.append(reward_rand)
        #             time_use_rand_list.append(time_use_rand)
        #             priority_sum_rand_list.append(priority_sum_rand)
        #             ns_prob_rand_list.append(ns_prob_rand)

        #             reward_greed, time_use_greed, priority_sum_greed, ns_prob_greed = start_greed(test_input, env)
        #             reward_greed_list.append(reward_greed)
        #             time_use_greed_list.append(time_use_greed)
        #             priority_sum_greed_list.append(priority_sum_greed)
        #             ns_prob_greed_list.append(ns_prob_greed)

        #             reward_multy, time_use_multy, priority_sum_multy, ns_prob_multy = start_multy(test_input, cfg)
        #             reward_multy_list.append(reward_multy)
        #             time_use_multy_list.append(time_use_multy)
        #             priority_sum_multy_list.append(priority_sum_multy)
        #             ns_prob_multy_list.append(ns_prob_multy)

        plt.legend(loc='upper left')  # 标签位置
        plt.rcParams['font.sans-serif'] = ['NotoSansCJK']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.rcParams['savefig.dpi'] = 600  # 图片像素
        plt.rcParams['figure.dpi'] = 600  # 分辨率

        fig = plt.figure()
        plt.plot(list(range(len(reward_list))), reward_list, c='red', label=u'指针网络', marker='o')
        plt.plot(list(range(len(reward_rand_list))), reward_rand_list, c='green', label=u'随机算法', marker='*')
        plt.plot(list(range(len(reward_greed_list))), reward_greed_list, c='blue', label=u'贪心算法', marker='>')
        plt.plot(list(range(len(reward_multy_list))), reward_multy_list, c='yellow', label=u'多级反馈队列算法', marker='D')
        plt.title(u"reward")
        plt.xlabel('服务请求个数(n)')
        plt.legend()
        fig.show()

        fig = plt.figure()
        plt.plot(list(range(len(time_use_list))), time_use_list, c='red', label=u'指针网络', marker='o')
        plt.plot(list(range(len(time_use_rand_list))), time_use_rand_list, c='green', label=u'随机算法', marker='*')
        plt.plot(list(range(len(time_use_greed_list))), time_use_greed_list, c='blue', label=u'贪心算法', marker='>')
        plt.plot(list(range(len(time_use_multy_list))), time_use_multy_list, c='yellow', label=u'多级反馈队列算法', marker='D')
        plt.title(u"reward1：运行时间")
        plt.xlabel('服务请求个数(n)')
        plt.legend()
        fig.show()

        fig = plt.figure()
        plt.plot(list(range(len(priority_sum_list))), priority_sum_list, c='red', label=u'指针网络', marker='o')
        plt.plot(list(range(len(priority_sum_rand_list))), priority_sum_rand_list, c='green', label=u'随机算法', marker='*')
        plt.plot(list(range(len(priority_sum_greed_list))), priority_sum_greed_list, c='blue', label=u'贪心算法',
                 marker='>')
        plt.plot(list(range(len(priority_sum_multy_list))), priority_sum_multy_list, c='yellow', label=u'多级反馈队列算法',
                 marker='D')
        plt.title(u"reward2：任务优先级")
        plt.xlabel('服务请求个数(n)')
        plt.legend()
        fig.show()

        fig = plt.figure()
        plt.plot(list(range(len(ns_prob_list))), ns_prob_list, c='red', label=u'指针网络', marker='o')
        plt.plot(list(range(len(ns_prob_rand_list))), ns_prob_rand_list, c='green', label=u'随机算法', marker='*')
        plt.plot(list(range(len(ns_prob_greed_list))), ns_prob_greed_list, c='blue', label=u'贪心算法', marker='>')
        plt.plot(list(range(len(ns_prob_multy_list))), ns_prob_multy_list, c='yellow', label=u'多级反馈队列算法', marker='D')
        plt.title(u"reward3：超时率")
        plt.xlabel('服务请求个数(n)')
        plt.legend()
        fig.show()

        print('ptr')
        print('综合效果', reward_list)
        print('目标1：运行时间', time_use_list)
        print('目标2：任务优先级', priority_sum_list)
        print('目标3：超时率', ns_prob_list)
        print('rand')
        print('综合效果', reward_rand_list)
        print('目标1：运行时间', time_use_rand_list)
        print('目标2：任务优先级', priority_sum_rand_list)
        print('目标3：超时率', ns_prob_rand_list)
        print('greed')
        print('综合效果', reward_greed_list)
        print('目标1：运行时间', time_use_greed_list)
        print('目标2：任务优先级', priority_sum_greed_list)
        print('目标3：超时率', ns_prob_greed_list)
        print('multy')
        print('综合效果', reward_multy_list)
        print('目标1：运行时间', time_use_multy_list)
        print('目标2：任务优先级', priority_sum_multy_list)
        print('目标3：超时率', ns_prob_multy_list)


    else:
        raise NotImplementedError('test only, specify test pkl file')
