# Pytorch Template

A template for pytorch project that is easy to use.

## Requirements

- pytorch
- tensorboard
- pyyaml

## Usage

### Configuration

Configurations are defined in yaml files.

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
└── trainers
    └── my_trainer.py
```

- Dataset should inherit from `BaseDataset` in folder `datasets`.
- Metrics is implemented as a child class of `BaseTrainer` in folder `trainers`.
- Loss is defined as a child class of `BaseLoss` in folder `losses`.
- Models inherits should be put in folder `models`

### Example

An example could be found in the `example1` folder.

This project is to train a model to fit the following function: 
$$x_1^2+e^{x_1+x_2}-2x_3$$

Use `python example1/run.py` to run the demo.

## TO DO ✅

- [ ] Add more examples.
