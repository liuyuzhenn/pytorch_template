# Pytorch Template

A template for pytorch project that is easy to use.

## Requirements

- pytorch
- tensorboard
- pyyaml

## Usage

### Configuration

Configurations are defined in yaml files, including:
- training hyperparameters, *e.g.* learning rate, epochs...
- model hyperparameters
- datset hyperparameters

See `./configs/example1.yml`.

### Project structure

The directory structure of a project should be like:
```
project
├── datasets
│   └── my_dataset.py
├── losses
│   └── my_loss.py
├── models
│   └── my_model.py
└── runners
    └── my_runner.py
```

- Dataset is defined as a child class of `BaseDataset` in folder `datasets`.
- Metrics is implemented in a child class of `BaseRunner` in folder `runners`.
- Loss is defined as a child class of `BaseLoss` in folder `losses`.
- Models should be put in folder `models`

### Example

An example could be found in the `example1` folder.

This project is to train a model to fit the following function: 
$$x_1^2+e^{x_1+x_2}-2x_3$$

Use `python example1/train.py` to run the demo.

## BUGS

Raising issues is welcomed.

## TO DO ✅

- [ ] Add more examples.

