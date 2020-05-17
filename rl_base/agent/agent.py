import abc


class Agent(abc.ABC):
    def __init__(self):
        self.memory = []

    @abc.abstractmethod
    def choose_action(self, state):
        pass

    def learn(self, state, action, reward, new_state=None):
        self.memory.append((state, action, reward))

    @abc.abstractmethod
    def conclude(self):
        pass
