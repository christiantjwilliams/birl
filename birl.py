import numpy as np
from algos import learn, policy
from env import LoopEnv
from env import MinecraftEnv
from utils import sample_demos, prob_dists
import argparse
import copy
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Inverse Reinforcement Learning')
    parser.add_argument('--policy', '-p', choices=('eps', 'bol'))
    parser.add_argument('--alpha', '-a', default=10, type=float, help='1/temperature of boltzmann distribution, '
                                                                      'larger value makes policy close to the greedy')
    parser.add_argument('--env_id', default=0, type=int)
    parser.add_argument('--r_max', default=1, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--n_iter', default=5000, type=int)
    parser.add_argument('--burn_in', default=1000, type=int)
    parser.add_argument('--dist', default='gaussian', type=str, choices=['uniform', 'gaussian', 'beta', 'gamma'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq):
    assert burn_in <= n_iter
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    return sampled_rewards


def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, **kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    env.rewards = sample_random_rewards(env.n_states, step_size, r_max)
    # step 2
    pi = learn.policy_iteration(env, gamma)
    # step 3
    for _ in tqdm(range(n_iter)):
        env_tilda = copy.deepcopy(env)
        env_tilda.rewards = mcmc_reward_step(env.rewards, step_size, r_max)
        q_pi_r_tilda = learn.compute_q_for_pi(env, pi, gamma)
        if is_not_optimal(q_pi_r_tilda, pi):
            pi_tilda = learn.policy_iteration(env_tilda, gamma, pi)
            if np.random.random() < compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
                env, pi = env_tilda, pi_tilda
        else:
            if np.random.random() < compute_ratio(demos, env_tilda, pi, env, pi, prior, alpha, gamma):
                env = env_tilda
        yield env.rewards


def is_not_optimal(q_values, pi):
    n_states, n_actions = q_values.shape
    for s in range(n_states):
        for a in range(n_actions):
            if q_values[s, pi[s]] < q_values[s, a]:
                return True
    return False


def compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
    ln_p_tilda = compute_posterior(demos, env_tilda, pi_tilda, prior, alpha, gamma)
    ln_p = compute_posterior(demos, env, pi, prior, alpha, gamma)
    ratio = np.exp(ln_p_tilda - ln_p)
    return ratio


def compute_posterior(demos, env, pi, prior, alpha, gamma):
    q = learn.compute_q_for_pi(env, pi, gamma)
    ln_p = np.sum([alpha * q[s, a] - np.log(np.sum(np.exp(alpha * q[s]))) for s, a in demos]) + np.log(prior(env.rewards))
    return ln_p


def mcmc_reward_step(rewards, step_size, r_max):
    new_rewards = copy.deepcopy(rewards)
    index = np.random.randint(len(rewards))
    step = np.random.choice([-step_size, step_size])
    new_rewards[index] += step
    new_rewards = np.clip(a=new_rewards, a_min=-r_max, a_max=r_max)
    if np.all(new_rewards == rewards):
        new_rewards[index] -= step
    assert np.any(rewards != new_rewards), \
        'rewards do not change: {}, {}'.format(new_rewards, rewards)
    return new_rewards


def sample_random_rewards(n_states, step_size, r_max):
    """
    sample random rewards form gridpoint(R^{n_states}/step_size).
    :param n_states:
    :param step_size:
    :param r_max:
    :return: sampled rewards
    """
    rewards = np.random.uniform(low=-r_max, high=r_max, size=n_states)
    # move these random rewards toward a gridpoint
    # add r_max to make mod to be always positive
    # add step_size for easier clipping
    rewards = rewards + r_max + step_size
    for i, reward in enumerate(rewards):
        mod = reward % step_size
        rewards[i] = reward - mod
    # subtracts added values from rewards
    rewards = rewards - (r_max + step_size)
    return rewards

def prepare_prior(dist, r_max):
    prior = getattr(prob_dists, dist[0].upper() + dist[1:] + 'Dist')
    if dist == 'uniform':
        return prior(xmax=r_max)
    elif dist == 'gaussian':
        return prior()
    elif dist in {'beta', 'gamma'}:
        return prior(loc=-r_max, scale=1/(2 * r_max))
    else:
        raise NotImplementedError('{} is not implemented.'.format(dist))

def main(args):
    np.random.seed(5)

    #prepare environments
    if args.env_id == 0:
        env_args = dict(loop_states=[1, 3, 2])
    else:
        assert args.env_id == 1, 'Invalid env id is given'
        env_args = dict(loop_states=[0, 3, 2])
    env_args['rewards'] = [0, 0, 0.7, 0.7]
    # env = LoopEnv(**env_args)
    env = MinecraftEnv('6by6_3_Z.csv', rewards={
			"wall": -1000,
			"fire": -0.2,
			"air": 0,
			"gravel": 0.2,
			"door": 0.2,
			"victim": 1,
			"victim-yellow": 1
		})

    # sample expert demonstrations
    # expert_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    # if args.policy == 'bol':
    #     expert_policy = policy.Boltzman(expert_q_values, args.alpha)
    #     print('pi \n', np.array([np.exp(args.alpha * expert_q_values[s])
    #                              / np.sum(np.exp(args.alpha * expert_q_values[s]), axis=-1) for s in env.states]))
    # else:
    #     expert_policy = policy.EpsilonGreedy(expert_q_values, epsilon=0.1)
    expert_policy = {"0": {"tile": 30, "heading": 0, "action": "turn_right"}, "1": {"tile": 30, "heading": 90, "action": "go_forth"}, "2": {"tile": 31, "heading": 90, "action": "turn_left"}, "3": {"tile": 31, "heading": 0, "action": "go_forth"}, "4": {"tile": 25, "heading": 0, "action": "go_forth"}, "5": {"tile": 19, "heading": 0, "action": "turn_right"}, "6": {"tile": 19, "heading": 90, "action": "go_forth"}, "7": {"tile": 20, "heading": 90, "action": "go_forth"}, "8": {"tile": 21, "heading": 90, "action": "go_forth"}, "9": {"tile": 22, "heading": 90, "action": "turn_right"}, "10": {"tile": 22, "heading": 180, "action": "go_forth"}, "11": {"tile": 28, "heading": 180, "action": "go_forth"}, "12": {"tile": 34, "heading": 180, "action": "turn_right"}, "13": {"tile": 34, "heading": 270, "action": "go_forth"}, "14": {"tile": 33, "heading": 270, "action": "turn_right"}, "15": {"tile": 33, "heading": 0, "action": "turn_right"}, "16": {"tile": 33, "heading": 90, "action": "go_forth"}, "17": {"tile": 34, "heading": 90, "action": "turn_left"}, "18": {"tile": 34, "heading": 0, "action": "go_forth"}, "19": {"tile": 28, "heading": 0, "action": "go_forth"}, "20": {"tile": 22, "heading": 0, "action": "go_forth"}, "21": {"tile": 16, "heading": 0, "action": "go_forth"}, "22": {"tile": 10, "heading": 0, "action": "turn_left"}, "23": {"tile": 10, "heading": 270, "action": "go_forth"}, "24": {"tile": 9, "heading": 270, "action": "turn_right"}, "25": {"tile": 9, "heading": 0, "action": "go_forth"}, "26": {"tile": 3, "heading": 0, "action": "turn_right"}, "27": {"tile": 3, "heading": 90, "action": "go_forth"}, "28": {"tile": 4, "heading": 90, "action": "go_forth"}}
    #expert_policy = {"0": {"tile": 658, "heading": 0, "action": "go_forth"}, "1": {"tile": 659, "heading": 0, "action": "go_forth"}, "2": {"tile": 660, "heading": 0, "action": "go_forth"}, "3": {"tile": 661, "heading": 0, "action": "go_forth"}, "4": {"tile": 661, "heading": 270, "action": "turn_right"}, "5": {"tile": 705, "heading": 270, "action": "go_forth"}, "6": {"tile": 746, "heading": 270, "action": "go_forth"}, "7": {"tile": 755, "heading": 270, "action": "go_forth"}, "8": {"tile": 772, "heading": 270, "action": "go_forth"}, "9": {"tile": 772, "heading": 0, "action": "turn_left"}, "10": {"tile": 773, "heading": 0, "action": "go_forth"}, "11": {"tile": 773, "heading": 270, "action": "turn_right"}, "12": {"tile": 773, "heading": 180, "action": "turn_right"}, "13": {"tile": 772, "heading": 180, "action": "go_forth"}, "14": {"tile": 772, "heading": 90, "action": "turn_right"}, "15": {"tile": 755, "heading": 90, "action": "go_forth"}, "16": {"tile": 746, "heading": 90, "action": "go_forth"}, "17": {"tile": 705, "heading": 90, "action": "go_forth"}, "18": {"tile": 661, "heading": 90, "action": "go_forth"}, "19": {"tile": 616, "heading": 90, "action": "go_forth"}, "20": {"tile": 571, "heading": 90, "action": "go_forth"}, "21": {"tile": 554, "heading": 90, "action": "go_forth"}, "22": {"tile": 530, "heading": 90, "action": "go_forth"}, "23": {"tile": 508, "heading": 90, "action": "go_forth"}, "24": {"tile": 475, "heading": 90, "action": "go_forth"}, "25": {"tile": 475, "heading": 0, "action": "turn_right"}, "26": {"tile": 476, "heading": 0, "action": "go_forth"}, "27": {"tile": 477, "heading": 0, "action": "go_forth"}, "28": {"tile": 477, "heading": 90, "action": "turn_left"}, "29": {"tile": 447, "heading": 90, "action": "go_forth"}, "30": {"tile": 416, "heading": 90, "action": "go_forth"}, "31": {"tile": 416, "heading": 180, "action": "turn_left"}, "32": {"tile": 416, "heading": 90, "action": "turn_right"}, "33": {"tile": 383, "heading": 90, "action": "go_forth"}, "34": {"tile": 355, "heading": 90, "action": "go_forth"}, "35": {"tile": 342, "heading": 90, "action": "go_forth"}, "36": {"tile": 313, "heading": 90, "action": "go_forth"}, "37": {"tile": 279, "heading": 90, "action": "go_forth"}, "38": {"tile": 247, "heading": 90, "action": "go_forth"}, "39": {"tile": 213, "heading": 90, "action": "go_forth"}, "40": {"tile": 185, "heading": 90, "action": "go_forth"}, "41": {"tile": 185, "heading": 180, "action": "turn_left"}, "42": {"tile": 184, "heading": 180, "action": "go_forth"}, "43": {"tile": 184, "heading": 270, "action": "turn_left"}, "44": {"tile": 184, "heading": 0, "action": "turn_left"}, "45": {"tile": 185, "heading": 0, "action": "go_forth"}, "46": {"tile": 185, "heading": 270, "action": "turn_right"}, "47": {"tile": 213, "heading": 270, "action": "go_forth"}, "48": {"tile": 247, "heading": 270, "action": "go_forth"}, "49": {"tile": 247, "heading": 180, "action": "turn_right"}, "50": {"tile": 246, "heading": 180, "action": "go_forth"}, "51": {"tile": 246, "heading": 270, "action": "turn_left"}, "52": {"tile": 246, "heading": 0, "action": "turn_left"}, "53": {"tile": 247, "heading": 0, "action": "go_forth"}, "54": {"tile": 247, "heading": 270, "action": "turn_right"}, "55": {"tile": 279, "heading": 270, "action": "go_forth"}, "56": {"tile": 313, "heading": 270, "action": "go_forth"}, "57": {"tile": 342, "heading": 270, "action": "go_forth"}, "58": {"tile": 355, "heading": 270, "action": "go_forth"}, "59": {"tile": 383, "heading": 270, "action": "go_forth"}, "60": {"tile": 416, "heading": 270, "action": "go_forth"}, "61": {"tile": 447, "heading": 270, "action": "go_forth"}, "62": {"tile": 477, "heading": 270, "action": "go_forth"}, "63": {"tile": 477, "heading": 180, "action": "turn_right"}, "64": {"tile": 476, "heading": 180, "action": "go_forth"}, "65": {"tile": 475, "heading": 180, "action": "go_forth"}, "66": {"tile": 475, "heading": 270, "action": "turn_left"}, "67": {"tile": 508, "heading": 270, "action": "go_forth"}, "68": {"tile": 530, "heading": 270, "action": "go_forth"}, "69": {"tile": 554, "heading": 270, "action": "go_forth"}, "70": {"tile": 554, "heading": 0, "action": "turn_left"}, "71": {"tile": 555, "heading": 0, "action": "go_forth"}, "72": {"tile": 556, "heading": 0, "action": "go_forth"}, "73": {"tile": 557, "heading": 0, "action": "go_forth"}, "74": {"tile": 557, "heading": 270, "action": "turn_right"}, "75": {"tile": 574, "heading": 270, "action": "go_forth"}, "76": {"tile": 574, "heading": 0, "action": "turn_left"}, "77": {"tile": 575, "heading": 0, "action": "go_forth"}, "78": {"tile": 576, "heading": 0, "action": "go_forth"}, "79": {"tile": 577, "heading": 0, "action": "go_forth"}, "80": {"tile": 578, "heading": 0, "action": "go_forth"}, "81": {"tile": 579, "heading": 0, "action": "go_forth"}, "82": {"tile": 580, "heading": 0, "action": "go_forth"}, "83": {"tile": 580, "heading": 270, "action": "turn_right"}, "84": {"tile": 580, "heading": 0, "action": "turn_left"}, "85": {"tile": 580, "heading": 270, "action": "turn_right"}, "86": {"tile": 580, "heading": 0, "action": "turn_left"}, "87": {"tile": 580, "heading": 270, "action": "turn_right"}, "88": {"tile": 580, "heading": 0, "action": "turn_left"}}
    demos = np.array(list(sample_demos(env, expert_policy)))
    # print('sub optimal actions {}/{}'.format(demos[:, 1].sum(), len(demos)))
    #assert np.all(expert_q_values[:, 0] > expert_q_values[:, 1]), 'a0 must be optimal action for all the states'

    # run birl
    start = time.time()
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(env, demos, step_size=0.05, n_iter=args.n_iter, r_max=args.r_max, prior=prior,
                                   alpha=args.alpha, gamma=args.gamma, burn_in=args.burn_in, sample_freq=1)
    end = time.time()
    print('total time:', end - start)
    # plot rewards
    # fig, ax = plt.subplots(1, env.n_states, sharey='all')
    # for i, axes in enumerate(ax.flatten()):
    #     axes.hist(sampled_rewards[:, i], range=(-args.r_max, args.r_max))
    # fig.suptitle('Loop Environment {}'.format(args.env_id), )
    # path = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-2], 'results',
    #                           'samples_env{}.png'.format(args.env_id))
    # plt.savefig(path)

    est_rewards = np.mean(sampled_rewards, axis=0)
    #est_rewards = parse_rewards(est_rewards)
    print('True rewards: ', env_args['rewards'])
    print('Estimated rewards: ', est_rewards)

    # compute optimal q values for estimated rewards
    env.rewards = est_rewards
    learner_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    for print_value in ('expert_q_values', 'learner_q_values'):
        print(print_value + '\n', locals()[print_value])
    print('Is a0 optimal action for all states: ', np.all(learner_q_values[:, 0] > learner_q_values[:, 1]))

def parse_rewards(rewards_matrix):
    new_rewards = []
    for i in range(len(rewards_matrix)):
        if i % 4 == 0:
            actual_reward = (rewards_matrix[i] + rewards_matrix[i+1] + rewards_matrix[i+2] + rewards_matrix[i+3])/4
            new_rewards.append(round(actual_reward, 2))
    return new_rewards

if __name__ == '__main__':
    args = get_args()
    main(args)
