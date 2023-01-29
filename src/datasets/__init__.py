def get_dataset(name):
    mod = __import__('datasets.{}'.format(name), fromlist=[''])
    return getattr(mod, _name_to_class(name))

def _name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))