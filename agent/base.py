class BaseAgent:
    def __init__(self, architecture, search_space, task_info):
        pass

    def train(self, train_data, valid_data, test_data, configs, save_model, save_history, path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass
