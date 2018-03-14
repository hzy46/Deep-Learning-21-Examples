from __future__ import print_function
import copy

MAP = \
    '''
.........
.       .
.     o .
.       .
.........
'''

# MAP = \
#     '''
# .........
# .  x    .
# .   x o .
# .       .
# .........
# '''
MAP = MAP.strip().split('\n')
MAP = [[c for c in line] for line in MAP]


DX = [-1, 1, 0, 0]
DY = [0, 0, -1, 1]


class Env(object):
    def __init__(self):
        self.map = copy.deepcopy(MAP)
        self.x = 1
        self.y = 1
        self.step = 0
        self.total_reward = 0
        self.is_end = False

    def interact(self, action):
        assert self.is_end is False
        new_x = self.x + DX[action]
        new_y = self.y + DY[action]
        new_pos_char = self.map[new_x][new_y]
        self.step += 1
        if new_pos_char == '.':
            reward = 0  # do not change position
        elif new_pos_char == ' ':
            self.x = new_x
            self.y = new_y
            reward = 0
        elif new_pos_char == 'o':
            self.x = new_x
            self.y = new_y
            self.map[new_x][new_y] = ' '  # update map
            self.is_end = True  # end
            reward = 100
        elif new_pos_char == 'x':
            self.x = new_x
            self.y = new_y
            self.map[new_x][new_y] = ' '  # update map
            reward = -5
        self.total_reward += reward
        return reward

    @property
    def state_num(self):
        rows = len(self.map)
        cols = len(self.map[0])
        return rows * cols

    @property
    def present_state(self):
        cols = len(self.map[0])
        return self.x * cols + self.y

    def print_map(self):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.x][self.y] = 'A'
        print('\n'.join([''.join([c for c in line]) for line in printed_map]))

    def print_map_with_reprint(self, output_list):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.x][self.y] = 'A'
        printed_list = [''.join([c for c in line]) for line in printed_map]
        for i, line in enumerate(printed_list):
            output_list[i] = line

