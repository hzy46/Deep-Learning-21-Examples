import numpy as np
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from gym import utils
from gym import spaces
from gym.envs.toy_text import discrete
from gym.envs.registration import register

MAPS = {
  "4x4": [
    "HHHD",
    "FFFF",
    "FHHH",
    "SHHH",
  ],
  "9x9": [
    "HHHHHHHHD",
    "HHHHHHHHF",
    "HHHHHHHHF",
    "HHHHHHHHF",
    "FFFFFFFFF",
    "FHHHHHHHH",
    "FHHHHHHHH",
    "FHHHHHHHH",
    "SHHHHHHHH",
  ],
}

class CorridorEnv(discrete.DiscreteEnv):
  """
  The surface is described using a grid like the following

    HHHD
    FFFF
    SHHH
    AHHH

  S : starting point, safe
  F : frozen surface, safe
  H : hole, fall to your doom
  A : adjacent goal
  D : distant goal

  The episode ends when you reach the goal or fall in a hole.
  You receive a reward of 1 if you reach the adjacent goal, 
  10 if you reach the distant goal, and zero otherwise.
  """
  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self, desc=None, map_name="9x9", n_actions=5):
    if desc is None and map_name is None:
      raise ValueError('Must provide either desc or map_name')
    elif desc is None:
      desc = MAPS[map_name]
    self.desc = desc = np.asarray(desc, dtype='c')
    self.nrow, self.ncol = nrow, ncol = desc.shape

    self.action_space = spaces.Discrete(n_actions)
    self.observation_space = spaces.Discrete(desc.size)

    n_state = nrow * ncol

    isd = (desc == 'S').ravel().astype('float64')
    isd /= isd.sum()

    P = {s : {a : [] for a in xrange(n_actions)} for s in xrange(n_state)}

    def to_s(row, col):
      return row*ncol + col
    def inc(row, col, a):
      if a == 0: # left
        col = max(col-1,0)
      elif a == 1: # down
        row = min(row+1, nrow-1)
      elif a == 2: # right
        col = min(col+1, ncol-1)
      elif a == 3: # up
        row = max(row-1, 0)

      return (row, col)

    for row in xrange(nrow):
      for col in xrange(ncol):
        s = to_s(row, col)
        for a in xrange(n_actions):
          li = P[s][a]
          newrow, newcol = inc(row, col, a)
          newstate = to_s(newrow, newcol)
          letter = desc[newrow, newcol]
          done = letter in 'DAH'
          rew = 1.0 if letter == 'A' \
              else 10.0 if letter == 'D' \
              else -1.0 if letter == 'H' \
              else 1.0 if (newrow != row or newcol != col) and letter == 'F' \
              else 0.0
          li.append((1.0/3.0, newstate, rew, done))

    super(CorridorEnv, self).__init__(nrow * ncol, n_actions, P, isd)

  def _render(self, mode='human', close=False):
    if close:
      return

    outfile = StringIO.StringIO() if mode == 'ansi' else sys.stdout

    row, col = self.s // self.ncol, self.s % self.ncol
    desc = self.desc.tolist()
    desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

    outfile.write("\n".join("".join(row) for row in desc)+"\n")
    if self.lastaction is not None:
      outfile.write("  ({})\n".format(self.get_action_meanings()[self.lastaction]))
    else:
      outfile.write("\n")

    return outfile

  def get_action_meanings(self):
    return [["Left", "Down", "Right", "Up"][i] if i < 4 else "NoOp" for i in xrange(self.action_space.n)]

register(
  id='CorridorSmall-v5',
  entry_point='environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '4x4',
    'n_actions': 5
  },
  timestep_limit=100,
)

register(
  id='CorridorSmall-v10',
  entry_point='environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '4x4',
    'n_actions': 10
  },
  timestep_limit=100,
)

register(
  id='CorridorBig-v5',
  entry_point='environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '9x9',
    'n_actions': 5
  },
  timestep_limit=100,
)

register(
  id='CorridorBig-v10',
  entry_point='environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '9x9',
    'n_actions': 10
  },
  timestep_limit=100,
)
