# Pytorch Template

An easy-to-use template for pytorch project.

## Requirements

- pytorch
- tensorboard
- hydra-core

## Usage

### Configuration

Configurations are defined in yaml files, including:
- training parameters, *e.g.* learning rate, epochs...
- model parameters
- datset parameters
- optimizer parameters
- lr_scheduler parameters

See `./configs/demo.yml`.

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

An example could be found in the `torch_temp` folder, which is to train a model to fit the following function: 
$$x_1^2+e^{x_1+x_2}-2x_3$$

Use `python train.py` to run the demo.

Or `CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh 4` for distributed training.
