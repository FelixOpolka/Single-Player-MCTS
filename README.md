# Single-Player Monte-Carlo Tree Search

General-purpose Python implementation of a **single-player** variant of the deep reinforcement learning **Monte-Carlo tree search** (MCTS) algorithm, originally introduced in the AlphaZero paper [1].

The algorithm builds on the idea of iteratively improving a deep policy network and a tree search operator in tandem. The policy improves its estimates by planning ahead via the tree search. Conversely, the tree search makes use of the policy to estimate the best branches to explore during the search.

The original AlphaZero algorithm was developed for two-player games. We modified the algorithm to play single-player games. Our implementation was adapted from the [minigo](https://github.com/tensorflow/minigo) implementation.

This repository provides an implementation of the MCTS algorithm that is independent of any deep learning framework. Furthermore, we provide a working example using PyTorch in which the agent learns to find the highest point on a map.

![](HillClimbing.gif)

## Files

The files `mcts.py`, and `static_env.py` provide the basic implementation and can be used independently of the application and the preferred deep learning framework. The training algorithm in `trainer.py` is largely application-independent but dependent on the choice of deep learning framework. We provide an example implementation using PyTorch. The remaining files are specific to the toy-example and need to be adapted for other applications.

## Requirements

### MCTS algorithm

* Numpy

### Toy example

* PyTorch
* OpenAI gym
* Matplotlib

## License

The implementation of the Monte-Carlo tree search algorithm in `mcts.py` was adapted from the [minigo](https://github.com/tensorflow/minigo) implementation of AlphaGo Zero, which is under the [Apache-2.0 license](https://github.com/FelixOpolka/Single-Player-MCTS/blob/master/minigo-license). Our changes are published under the MIT license.

## Contributors

This repository was part of a Bachelor Thesis project by Felix Opolka supervised by Vladimir Golkov and Prof Daniel Cremers. Please cite [to be announced] if you use this code in your work.

## References

<!-- [0] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis. "Mastering the game of Go with deep neural networks and tree search." In: Nature 529 (2016), pp. 484–489. -->

<!-- [0] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap, F. Hui, L. Sifre, G. van den Driessche, T. Graepel, and D. Hassabis. “Mastering the game of Go without human knowledge.” In: Nature 550.7676 (2017), pp. 354–359. -->

[1] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, and D. Hassabis. “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.” In: arXiv:1712.01815 [cs] (2017).
