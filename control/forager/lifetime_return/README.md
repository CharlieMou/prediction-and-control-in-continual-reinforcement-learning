# TwoBiomeForager historical-task lifetime return

`DQN.py`, `dqn_gac.py`, and `DQN_online_ewc.py` evaluate the reward mapping
opposite to the current training mapping. Evaluation starts only after that
mapping has previously been completed. It runs in independent environments
whose reward mapping is fixed, so it neither advances the training environment
nor changes replay, optimizer, target-network, GAC, or EWC state.

The default diagnostic uses five 10,000-step near-greedy continuing rollouts
every 50,000 training steps. Its W&B keys are under `lifetime_return/`.
The same metric values and W&B step are written to one CSV per run in this
directory. Filenames contain the method, readable principal parameters, seed,
evaluation settings, and a short seed-independent hash of all functional
command-line settings.

Relevant options are:

```text
--lifetime-eval-frequency 50000
--lifetime-eval-steps 10000
--lifetime-eval-rollouts 5
--lifetime-eval-epsilon 0.001
--lifetime-output-dir control/forager/lifetime_return
```

The evaluation frequency must be divisible by `--log-frequency`, allowing the
training and lifetime metrics to be committed to W&B in one call at one step.

After all methods and seeds finish, create the aggregate two-panel figure with:

```bash
python control/forager/lifetime_return/plot_lifetime_return.py
```

This reads every `*_lifetime_return.csv`, averages only matching configurations
over seeds, draws 95% normal confidence intervals, and writes:

- `figs/two_biome_lifetime_return.pdf`
- `figs/two_biome_lifetime_return.png`
