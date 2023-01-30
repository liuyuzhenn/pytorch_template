def get_trainer(name):
    mod = __import__('trainer.{}'.format(name), fromlist=[''])
    return getattr(mod, _name_to_class(name))

def _name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))
