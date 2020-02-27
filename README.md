# aurora

Intuitive, high-level training framework for research and development. It abstracts away a lot of boilerplate normally associated with training and evaluating PyTorch models, without limiting your flexibility. Aurora provides the following:

* A way to specify training runs at a high level, while having fine-grained control over parts of the training
* Automated checkpointing, logging and resuming of runs
* A [sacred](https://github.com/IDSIA/sacred) inspired configuration management
* Reproducibility by keeping track of configuration, code and random state of each run

## Installation

Make sure you have numpy and torch installed, then install with pip:

```bash
pip install git+https://github.com/alebeck/aurora
```

## Getting started

A minimal working example can look like that:

##### config.yaml
```yaml
data:
  __type__: torchvision.datasets.MNIST
  root: ./data
  download: true

model:
  __type__: torchvision.models.vgg16
  pretrained: true

optimizer:
  __type__: torch.optim.Adam

loss_fn:
  __type__: torch.nn.CrossEntropyLoss

log_path: ./logs_mnist
batch_size: 16
epochs: 100
```

##### train.py
```python
from aurora import Run

run = Run()
run.add_config('./config.yaml')

run()
```

Aurora will now take care of setting up data set and model and create a log directory inside `./logs_mnis`. There it will save the configuration, checkpoints, log files as well as training metrics. During the training, the user is constantly informed about training progress through the command line.


## Customizing parts of your training
Of course, training your model is not always as straight-forward as depicted in the example above. You could, for example, want to have finer control about what happens during each training step. This can be accomplished using aurora's decorators:

```python
@run.train_step
@run.validate_step
def step(_batch, _model, _optimizer, _is_validate):
    x, y = _batch
    y_hat = _model(x)
    loss = (y - y_hat) ** 2

    if not _is_validate:
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()
```

These lines of code register the `step` function as a handler for both training and validation steps. In the functions' parameter list, you can specify which things you need for the function and aurora will automatically inject them based on name. Parameters starting with `_` are so-called built-in parameters which will inject special values:

* `_batch`: current mini-batch
* `_model`: the model
* `_optimizer`: the optimizer
* `_is_validate`: `True` iff we're currently in the validation step. Useful when a function is registered for both train and validation.
* `_loss_fn`
* `_epoch`: current epoch
* `_log_path`: the directory where logs are written to
* `_is_cuda`: are we using CUDA for training?

In addition to these built-ins, all keys from your configuration can be used as parameter names and the corresponding values will be injected automatically.

## Configuration management

Configurating your training runs is very intuitive with aurora. You have already seen one example of how to declare configurations: By specifying yaml files using `run.add_config()`. You can add as many configuration files to you run as you want. In addition, you can define or overwrite configuration values using the command line args in a fashion inspired by [sacred](https://github.com/IDSIA/sacred):

```bash
python train.py with model.pretrained=False important_classes=[0,1,2,3]
```

This will change the value of `model.pretrained` from `True` (as defined in `config.yaml`) to `False` and will define a new config called `important_classes` with the value of `[0,1,2,3]`. You could use this value as follows:

```python
@run.train_step
@run.validate_step
def step(_batch, _model, _loss_fn, important_classes):
    x, y = _batch
    y_hat = _model(x)
    loss = _loss_fn(y_hat, y)
    if y_hat.argmax() in important_classes:
        loss = 2 * loss
    ...
```

## Available decorators