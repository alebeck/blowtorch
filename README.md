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
`Run.run()` takes following options:
* `model`: `torch.nn.Module`
* `train_loader`: `torch.utils.data.DataLoader`
* `val_loader`: `torch.utils.data.DataLoader`
* `loggers`: `Optional[List[aurora.logging.BaseLogger]]` (List of loggers that subscribe to various logging events, see logging section)
* `max_epochs`: `int` (default `1`)
* `use_gpu`: `bool` (default `True`)
* `gpu_id`: `int` (default `0`)
* `resume`: `Optional[Union[str, pathlib.Path]]` (Path to checkpoint to resume training from, default `None`)
* `save_path`: `Union[str, pathlib.Path]` (Path to directory that blowtorch will save logs and checkpoints to, default `'train_logs'`)
* `run_name`: `Optional[str]` (Name associated with that run, will be randomly created if None, default `None`)
* `optimize_metric`: `Optional[str]` (train metric that will be used for optimization, will pick the first returned one if None, default `None`)
* `checkpoint_metric`: `Optional[str]` (validation metric that will be used for checkpointing, will pick the first returned one if None, default `None`)
* `smaller_is_better`: `bool` (default `True`)
* `optimize_first`: `bool` (whether optimization should occur during the first epoch, default `False`)
* `detect_anomalies`: `bool` (enable autograd anomaly detection, default `False`)

## Logging
Blowtorch will create a folder with name "<timestamp>-<name>-<sequential integer>" for each run inside the `save_path` directory. Here it will save:
* `config.yaml` containing all configuration values
* `log.txt` listing all metrics for each epoch
* `model-summary.txt`: Summary of the model architecture
* `source.txt`: Source code of the model as well as of all decorator functions
* `checkpoints`: Directory containing checkpoints, each consisting of model & optimizer state and epoch information.

Additional loggers can be added through `Run`s `loggers` parameter. Blowtorch comes with a `blowtorch.loggers.WandbLogger` and a `blowtorch.loggers.TensorBoardLogger`. Custom loggers can be created by subclassing `blowtorch.loggers.BaseLogger`.

## Decorators
Signature of all decorator functions

## Reproduceability

## Learning rate schedulers