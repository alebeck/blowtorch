# blowtorch

Intuitive, high-level PyTorch training framework for research and development. Abstracting away boilerplate code normally associated with training and evaluating PyTorch models, with a more intuitive API then similar frameworks. Blowtorch provides:

* A way to specify training runs at high level, while not giving up on fine-grained control
* Automated checkpointing, logging and resuming of runs
* A [sacred](https://github.com/IDSIA/sacred) inspired configuration management
* Reproducibility by keeping track of configuration, code and random state of each run

## Installation
Make sure you have `numpy` and `torch` installed, then install with pip:

```shell script
pip install --upgrade blowtorch
```

## Example
```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import vgg16
from blowtorch import Run

run = Run(random_seed=123)

train_loader = DataLoader(CIFAR10('.', train=True, download=True, transform=ToTensor()))
val_loader = DataLoader(CIFAR10('.', train=False, download=True, transform=ToTensor()))


@run.train_step(train_loader)
@run.validate_step(val_loader)
def step(batch, model):
    x, y = batch
    y_hat = model(x)
    loss = (y - y_hat).square().mean()
    return loss

# will be called when model has been moved to the desired device 
@run.configure_optimizers
def configure_optimizers(model):
    return Adam(model.parameters())


run(vgg16(num_classes=10), num_epochs=100)
```

Note that in the above example, blowtorch automatically takes care of moving the model and tensors to the appropriate devices. It automatically sets the model to train/eval mode and activates/deactivates gradient calculation, respectively. Furthermore, logging and checkpointing is taken care of automatically.

## Features

### Configuration management
You can pass multiple configuration files in YAML format to your `Run`, e.g.
```python
run = Run(config_files=['config/default.yaml'])
```
Configuration values can then be accessed via e.g. `run['model']['num_layers']`. Dotted notation is also supported, e.g. `run['model.num_layers']`.  When executing your training script, individual configuration values can be overwritten as follows:

```shell script
python train.py with model.num_layers=4 model.use_dropout=True
```

### Automatic argument injection
If you need the epoch number or torch device (or other things) in your decorated functions, just specify them. Blowtorch will analyze the function signature and inject the requested arguments automatically:

```python
@run.train_step(train_loader)
def train_step(batch, model, epoch, device):
    ...
```

Supported arguments for all decorated functions can be found in the respective docstrings.

### Logging
Blowtorch will create a folder with name "[timestamp]-[name]-[sequential integer]" for each run inside the `run.save_path` directory. Here it will save the runs's configuration, metrics, a model summary, checkoints as well as source code. Additional loggers can be added through `Run`s `loggers` parameter:

* `blowtorch.loggers.WandbLogger`: Logs to Weights & Biases
* `blowtorch.loggers.TensorBoardLogger`: Logs to TensorBoard

Custom loggers can be created by subclassing `blowtorch.loggers.BaseLogger`.

### Flexible validation & checkpointing
Just specify when and how often you would like to validate using a natural-language-like syntax:

```python
@run.val_step(val_loader, every=2, at=0)
def val_step(batch, model):
    ...
```

Of course, you can use multiple validation functions which different data loaders and validation preferences, Blowtorch will join the metrics conveniently.

### Hooks
TODO

### Distributed data parallel
DDP should work but is still experimental. Just add the `ddp` decorator to your ...

### Run options
`Run.run()` takes following options:
* `model`: `torch.nn.Module`
* `train_loader`: `torch.utils.data.DataLoader`
* `val_loader`: `torch.utils.data.DataLoader`
* `loggers`: `Optional[List[aurora.logging.BaseLogger]]` (List of loggers that subscribe to various logging events, see logging section)
* `max_epochs`: `int` (default `1`)
* `use_gpu`: `bool` (default `True`)
* `gpu_id`: `int` (default `0`)
* `resume_checkpoint`: `Optional[Union[str, pathlib.Path]]` (Path to checkpoint directory to resume training from, default `None`)
* `save_path`: `Union[str, pathlib.Path]` (Path to directory that blowtorch will save logs and checkpoints to, default `'train_logs'`)
* `run_name`: `Optional[str]` (Name associated with that run, will be randomly created if None, default `None`)
* `optimize_metric`: `Optional[str]` (train metric that will be used for optimization, will pick the first returned one if None, default `None`)
* `checkpoint_metric`: `Optional[str]` (validation metric that will be used for checkpointing, will pick the first returned one if None, default `None`)
* `smaller_is_better`: `bool` (default `True`)
* `optimize_first`: `bool` (whether optimization should occur during the first epoch, default `False`)
* `detect_anomalies`: `bool` (enable autograd anomaly detection, default `False`)
