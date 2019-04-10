# Neural Architecture Search for Multi-task Neural Networks

## Introduction

Train a one-shot neural architecture search (NAS) agent that can design multi-task models.

## Usage

### Train

```
python main.py --train
```

Arguments:

 * `--type`: (default: `1`)
   * `1`: Train a single task single objective NAS agent for task *i* model.
   * `2`: Train a multi-task single objective NAS agent.
   * `3`: Train a single task multi-objective NAS agent for task *i* model.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
   * `2`: Omniglot
 * `--task`: Task ID (for type `1`) (default: None)
 * `--save`: A flag used to decide whether to save model or not.
 * `--save_history`: A flag used to decide whether to save accuracy history or not.
 * `--load`: Load a pre-trained model before training.
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.

### Evaluate

```
python main.py --eval
```

Arguments:

 * `--type`: (default: `1`)
   * `1`: Evaluate a single task single objective NAS agent for task *i* model.
   * `2`: Evaluate a multi-task single objective NAS agent.
   * `3`: Evaluate a single task multi-objective NAS agent for task *i* model.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
   * `2`: Omniglot
 * `--task`: Task ID (for type `1`) (default: None)
 * `--id`: Evaluate *id* final model. (default: None)
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)

### Final Train

```
python main.py --final
```

Arguments:

 * `--type`: (default: `1`)
   * `3`: Train a single task multi-objective NAS agent for task *i* model.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
   * `2`: Omniglot
 * `--task`: Task ID (for type `1`) (default: None)
 * `--id`: Train *id* final model found in search phase. (default: None)
 * `--save`: A flag used to decide whether to save model or not.
 * `--save_history`: A flag used to decide whether to save accuracy history or not.
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.
