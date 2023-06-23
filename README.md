# afdrl
Scalable Asynchronous Federated Deep Reinforcement Learning Simulator

# about
AFDRL simulates a large-scale deterministic federation of clients seeking to maximize reward in an ALE game.

# building
CMake is required to build afdrl.
```bash
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make
    $ afdrl/afdrl
```

# usage
To run a simulation of 64 participants, execute the following:
```
  $ afdrl/afdrl -c 64 --gpu
```
