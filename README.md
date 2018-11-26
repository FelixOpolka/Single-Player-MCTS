# Single-Player Monte-Carlo Tree Search

General-purpose Python implementation of a **single-player** variant of the **Monte-Carlo tree search** (MCTS) algorithm for deep reinforcement learning. The original two-player variant was introduced in the [AlphaZero paper](https://arxiv.org/abs/1712.01815) by Silver et al.

The algorithm builds on the idea of iteratively improving a deep policy network and a tree search in tandem. The policy improves its estimates by planning ahead via the tree search. Conversely, the tree search makes use of the progressively more accurate policy to estimate the best branches to explore during the search.

The original AlphaZero algorithm was developed for two-player games. We modified the algorithm to play single-player games. Our implementation was adapted from the [minigo](https://github.com/tensorflow/minigo) implementation, which is limited to the game of Go (two-player game).

This repository provides an implementation of the MCTS algorithm that is independent of any deep learning framework. Furthermore, we provide a working example using PyTorch in which the agent learns to find the highest point on a map.

A fast C++ version will be coming soon!

![](HillClimbing.gif)


## Files

The files `mcts.py`, and `static_env.py` provide the basic implementation and can be used independently of the application and the preferred deep learning framework. The training algorithm in `trainer.py` is largely application-independent but dependent on the choice of deep learning framework. We provide an example implementation using PyTorch. The remaining files are specific to the toy-example and need to be adapted for other applications.

## Requirements

### MCTS algorithm

* NumPy

### Toy example

* PyTorch
* OpenAI gym
* Matplotlib

## License

The implementation of the Monte-Carlo tree search algorithm in `mcts.py` was adapted from the [minigo](https://github.com/tensorflow/minigo) implementation of AlphaGo Zero, which is under the [Apache-2.0 license](https://github.com/FelixOpolka/Single-Player-MCTS/blob/master/minigo-license). Our changes are published under the MIT license.

## Contributors

This repository was part of a Bachelor Thesis project by Felix Opolka supervised by Vladimir Golkov and Prof. Daniel Cremers. 

## How to Cite

Please cite [to be announced] if you use this code in your work.
