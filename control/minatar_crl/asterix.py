"""Compatibility adapter for the MinAtar Asterix environment."""

import numpy as np
from minatar.environments.asterix import Env as MinAtarAsterixEnv


class Env(MinAtarAsterixEnv):
	def __init__(self, ramping=True, random_state=None,
				use_minimal_observation=True):
		super().__init__(ramping=ramping)
		self.random = random_state if random_state is not None \
				else np.random.RandomState()
		if not use_minimal_observation:
			for channel in range(len(self.channels), 10):
				self.channels[f"misc_{channel}"] = channel
		self.reset()
