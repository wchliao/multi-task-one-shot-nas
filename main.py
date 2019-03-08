import argparse
import yaml
from namedtuple import TaskInfo, DataConfigs, PretrainConfigs, ControllerConfigs, FinalModelConfigs, Configs, LayerArguments, OperationArguments
from data_loader import CIFAR100Loader
from agent import SingleTaskSingleObjectiveAgent
from agent import MultiTaskSingleObjectiveSingleModelAgent
from agent import MultiTaskSingleObjectiveMultiModelAgent


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--type', type=int, default=1, help='1: Single task single objective\n'
                                                            '2: Multi-task single objective single model\n'
                                                            '3: Multi-task single objective multi model')
    parser.add_argument('--data', type=int, default=1, help='1: CIFAR-100')
    parser.add_argument('--task', type=int, default=None)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_history', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--path', type=str, default='saved_models/default/')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    configs = _load_configs()
    architecture = _load_architecture()
    search_space = _load_search_space()

    if args.data == 1:
        train_data = CIFAR100Loader(batch_size=configs.data.batch_size, type='train', drop_last=True)
        valid_data = CIFAR100Loader(batch_size=configs.data.batch_size, type='valid', drop_last=False)
        test_data = CIFAR100Loader(batch_size=configs.data.batch_size, type='test', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(train_data.num_classes)

    if args.type == 1:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes[args.task],
                             num_channels=train_data.num_channels,
                             num_tasks=1
                             )

        train_data = train_data.get_loader(args.task)
        valid_data = valid_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)

    else:
        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes,
                             num_channels=train_data.num_channels,
                             num_tasks=num_tasks
                             )

    if args.type == 1:
        agent = SingleTaskSingleObjectiveAgent(architecture, search_space, task_info)
    elif args.type == 2:
        agent = MultiTaskSingleObjectiveSingleModelAgent(architecture, search_space, task_info)
    elif args.type == 3:
        agent = MultiTaskSingleObjectiveMultiModelAgent(architecture, search_space, task_info)
    else:
        raise ValueError('Unknown setting: {}'.format(args.type))

    if args.load:
        agent.load(args.path)

    agent.train(train_data=train_data,
                valid_data=valid_data,
                test_data=test_data,
                configs=configs,
                save_model=args.save,
                save_history=args.save_history,
                path=args.path,
                verbose=args.verbose
                )


def evaluate(args):
    configs = _load_configs()
    architecture = _load_architecture()
    search_space = _load_search_space()

    if args.data == 1:
        data = CIFAR100Loader(batch_size=configs.data.batch_size, type='test', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(data.num_classes)

    if args.type == 1:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        task_info = TaskInfo(image_size=data.image_size,
                             num_classes=data.num_classes[args.task],
                             num_channels=data.num_channels,
                             num_tasks=1
                             )

        data = data.get_loader(args.task)

    else:
        task_info = TaskInfo(image_size=data.image_size,
                             num_classes=data.num_classes,
                             num_channels=data.num_channels,
                             num_tasks=num_tasks
                             )

    if args.type == 1:
        agent = SingleTaskSingleObjectiveAgent(architecture, search_space, task_info)
    elif args.type == 2:
        agent = MultiTaskSingleObjectiveSingleModelAgent(architecture, search_space, task_info)
    elif args.type == 3:
        agent = MultiTaskSingleObjectiveMultiModelAgent(architecture, search_space, task_info)
    else:
        raise ValueError('Unknown setting: {}'.format(args.type))

    agent.load(args.path)
    accuracy, model_size = agent.eval(data)

    print('Accuracy: {}'.format(accuracy))
    print('Model size: {}'.format(model_size))


def _load_configs():
    with open('configs/train.yaml', 'r') as f:
        configs = yaml.load(f)

    data_configs = DataConfigs(**configs['data'])
    pretrain_configs = PretrainConfigs(**configs['pretrain'])
    controller_configs = ControllerConfigs(**configs['controller'])
    final_configs = FinalModelConfigs(**configs['final'])

    return Configs(data=data_configs, pretrain=pretrain_configs, controller=controller_configs, final=final_configs)


def _load_architecture():
    with open('configs/architecture.yaml', 'r') as f:
        configs = yaml.load(f)

    return [LayerArguments(**config) for config in configs]


def _load_search_space():
    with open('configs/search_space.yaml', 'r') as f:
        configs = yaml.load(f)

    return [OperationArguments(**layer) for layer in configs]


def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        print('No flag is assigned. Please assign either \'--train\' or \'--eval\'.')


if __name__ == '__main__':
    main()