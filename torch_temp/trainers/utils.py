import numpy as np
import torch
import torchvision.utils as vutils


def recursive(func):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], list):
            return [wrapper(*[x, *args[1:]], **kwargs) for x in args[0]]
        elif isinstance(args[0], tuple):
            return tuple([wrapper(*[x, *args[1:]], **kwargs) for x in args[0]])
        elif isinstance(args[0], dict):
            return {k: wrapper(*[v, *args[1:]], **kwargs) for k, v in args[0].items()}
        else:
            return func(*args, **kwargs)
    return wrapper


@recursive
def to_device(sample, device):
    if device == 'cpu':
        if isinstance(sample, torch.Tensor):
            return sample.cpu()
        else:
            return sample
    elif device == 'cuda':
        if isinstance(sample, torch.Tensor):
            return sample.cuda()
        else:
            return sample
    else:
        return sample.to(device)


@recursive
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError(
            "invalid input type {} for tensor2float".format(type(vars)))


@recursive
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError(
            "invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(writer, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            writer.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                writer.add_scalar(name, value[idx], global_step)


def save_images(writer, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError(
                "invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            writer.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                writer.add_image(name, preprocess(
                    name, value[idx]), global_step)


def dict_to_str(d, sep='| '):
    out = []
    for k, v in d.items():
        out.append('{}: {} '.format(k, v))
    return sep.join(out)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError(
                        "Invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError(
                        "Invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        if self.count > 0:
            return {k: v / self.count for k, v in self.data.items()}
        else:
            return {}
