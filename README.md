# blowtorch

Intuitive, high-level training framework for research and development. It abstracts away boilerplate normally associated with training and evaluating PyTorch models, without limiting your flexibility. Blowtorch provides the following:

* A way to specify training runs at a high level, while not giving up on fine-grained control over individual training parts
* Automated checkpointing, logging and resuming of runs
* A [sacred](https://github.com/IDSIA/sacred) inspired configuration management
* Reproducibility by keeping track of configuration, code and random state of each run

## Installation
Make sure you have `numpy` and `torch` installed, then install with pip:

```shell script
pip install --upgrade blowtorch
```

## Minimal working example
```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.datasets import ImageNet
from blowtorch import Run

run = Run(random_seed=123)

@run.train_step
@run.validate_step
def step(batch, model):
    x, y = batch
    y_hat = model(x)
    loss = (y - y_hat) ** 2
    return loss

# will be called when model has been moved to the desired device 
@run.configure_optimizers
def configure_optimizers(model):
    return Adam(model.parameters())

train_loader = DataLoader(ImageNet('.', split='train'), batch_size=4)
val_loader = DataLoader(ImageNet('.', split='val'), batch_size=4)

run(vgg16(), train_loader, val_loader)
```

## Configuration
You can pass multiple configuration files in YAML format to your `Run`, e.g.
```python
run = Run(config_files=['config/default.yaml'])
```
Configuration values can then be accessed via e.g. `run['model']['num_layers']`. Dotted notation is also supported, e.g. `run['model.num_layers']`.  When executing your training script, individual configuration values can be overwritten as follows:

```shell script
python train.py with model.num_layers=4 model.use_dropout=True
```

## Run options
...

## Decorators
Signature of all decorator functions

## Logging
explain logging behavior and additional loggers (wandb + how to write own)

## Reproduceability

## Learning rate schedulers