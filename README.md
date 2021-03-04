# blowtorch

Intuitive, high-level training framework for research and development. It abstracts away lots of boilerplate normally associated with training and evaluating PyTorch models, without limiting your flexibility. Aurora provides the following:

* A way to specify training runs at a high level, while having fine-grained control over individual parts of the training
* Automated checkpointing, logging and resuming of runs
* A [sacred](https://github.com/IDSIA/sacred) inspired configuration management
* Reproducibility by keeping track of configuration, code and random state of each run

## Installation