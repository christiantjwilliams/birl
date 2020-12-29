import numpy as np
import csv
import os
from os.path import join, isfile

class MinecraftEnv:
    def __init__(self, csv_map, rewards=None, loop_states=None, trans_probs=None):
        self.world_grid = self.get_grid(csv_map)
        self.n_states = len(self.world_grid)*len(self.world_grid[0])*4
        self.n_actions = 3
        self.states = np.array([*range(self.n_states)], dtype=int)
        self.loop_states = loop_states
        self.trans_probs = trans_probs if trans_probs else self._get_trans_probs()
        self.state = None
        self._rewards = self.set_rewards(self.get_grid(csv_map), rewards)

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        if isinstance(rewards, list):
            assert len(rewards) == self.n_states, 'Invalid rewards specified'
            rewards = np.array(rewards)
        assert rewards.shape == (self.n_states,), 'Invalid rewards specified'
        self._rewards = rewards

    def get_grid(self, csv_map):
        with open(join('maps', csv_map), encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            world_grid = []
            for row in csv_reader:
                world_grid.append(row)
            return world_grid

    def set_rewards(self, world_grid, rewards_dict):
        rewards = []
        map_tile_to_fullname = {'':'air','W':'wall','F':'fire','V':'victim', 'VV' : 'victim-yellow',
                'D':'door','G':'gravel', 'S':'air'}
        for row in world_grid:
            for tile in row:
                rewards.append(rewards_dict[map_tile_to_fullname[tile]])
        return rewards

    def step(self, a):
        # breakpoint()
        assert 0 <= a < self.n_actions, '{} is invalid action index. ' \
                                        'Action must be in the range of [0, {}]'.format(a, self.n_actions)
        self.state = np.random.choice(self.states, p=self.trans_probs[self.state, a])
        reward = self._get_reward()
        return self.state, reward

    def reset(self):
        #start at state S
        state_counter = 0
        final_counter = 0
        for i in range(len(self.world_grid)):
            for j in range(len(self.world_grid[0])):
                if self.world_grid[i][j] == 'S':
                    final_counter = state_counter
                else:
                    state_counter += 1
        self.state = final_counter
        print('starting state:', self.state)
        return self.state

    def _get_reward(self, state=None):
        assert self.rewards is not None, 'rewards is not specified'
        state = self.state if state is None else state
        return self.rewards[state]

    def _get_trans_probs(self):
        trans_probs = np.full((self.n_states, self.n_actions, self.n_states), 1,)
        return trans_probs


if __name__ == '__main__':
    # The expert in this environments loops s1, s2, s3
    trans_probs = np.empty(shape=(4, 2, 4), dtype=np.float32)
    loop_states = [1, 3, 2]
    a1_next_state = [s for s in range(trans_probs.shape[0]) if s not in loop_states][0]
    trans_probs[:, 1] = np.eye(4)[a1_next_state]
    for state in range(trans_probs.shape[0]):
        trans_probs[a1_next_state, 0, state] = 0 if state == a1_next_state else 1/3
    for state, a0_next_state in zip(loop_states, loop_states[1:] + [loop_states[0]]):
        trans_probs[state, 0] = np.eye(4)[a0_next_state]

    env = LoopEnv(rewards=[0, 0, 0, 1], loop_states=loop_states)
    obs = env.reset()
    for _ in range(100):
        a = np.random.randint(env.n_actions)
        obs, reward = env.step(a)
        print('obs: {}, action: {}, reward: {}'.format(obs, a, reward))