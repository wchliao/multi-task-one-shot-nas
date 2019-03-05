import yaml
from collections import namedtuple


# Named tuples for configurations

with open('configs/train.yaml', 'r') as f:
    _configs = yaml.load(f)

DataConfigs = namedtuple('DataConfigs', _configs['data'].keys())
PretrainConfigs = namedtuple('PretrainConfigs', _configs['pretrain'].keys())
ControllerConfigs = namedtuple('ControllerConfigs', _configs['controller'].keys())
FinalModelConfigs = namedtuple('FinalModelConfigs', _configs['final'].keys())
Configs = namedtuple('Configs', ['data', 'pretrain', 'controller', 'final'])

with open('configs/architecture.yaml', 'r') as f:
    _configs = yaml.load(f)

LayerArguments = namedtuple('LayerArguments', _configs[0].keys())

with open('configs/search_space.yaml', 'r') as f:
    _configs = yaml.load(f)

OperationArguments = namedtuple('OperationArguments', _configs[0].keys())


# Others

TaskInfo = namedtuple('TaskInfo', ['image_size', 'num_classes', 'num_channels', 'num_tasks'])
