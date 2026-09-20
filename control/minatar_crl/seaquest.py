"""Compatibility adapter for the MinAtar Seaquest environment."""

import numpy as np
from minatar.environments.seaquest import Env as MinAtarSeaquestEnv


class Env(MinAtarSeaquestEnv):
	def __init__(self, ramping=True, random_state=None,
				use_minimal_observation=True):
		super().__init__(ramping=ramping)
		self.random = random_state if random_state is not None \
				else np.random.RandomState()
		self.reset()
