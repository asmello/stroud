# stroud

Stroud is a Python library that implements different models for dealing
with a special class of Reinforcement Learning (RL) problems.

In particular, the kind of problems it tackles have the following properties:

0. We wish to maximize cumulative reward;
1. Feedback is slow and expensive;
2. The reward varies depending on a single choice, and can be estimated given some context;
3. The choice is not defined by a discrete set of actions, but by a continuous parameter $p$.

This can be thought of as a variation on the contextual multi-armed bandit formulation, where instead of deciding which arm to pull, we decide how much.

## Dependencies

* Numpy
* Tensorflow

## Installation

TODO (not yet a proper python package)

## Contextualization

Within this framework, we generally have a more or less fixed set of entities we wish to optimize for -- say, a set of factories with different attributes --, so while we may have a continuous features space, we only have a limited amount of context instances to work with.

In contrast with more general RL problems, the system is assumed to be stateless - one experiment does not influence the outcome of the next one. This makes the reward a pure function of the context and the chosen parameter, though it need not be fully deterministic. Approximating this function from a set of (potentially noisy) samples is something trivial for a deep Neural Network, but obtaining enough samples for that can be prohibitively costly (property 1).

This library focuses on offering a few different means toward simultaneously learning the reward function and maximizing the accumulated reward.

## What's implemented

For now, only a single model is implemented - a multi-headed neural network that learns through a variation on Thompson Sampling. More models will come in the future.
