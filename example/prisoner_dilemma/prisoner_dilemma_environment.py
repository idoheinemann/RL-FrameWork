import numpy as np

from rl_base.agent.agent import Agent
from rl_base.env.environment import Environment, Reward, State


class PrisonerDilemmaReward:
    COST_TABLE = (0.25, 0.0, 1.0, 0.05)

    def __init__(self, env, idx, first_shift, second_shift):
        self.idx = idx
        self.first_shift = first_shift
        self.second_shift = second_shift
        self.env = env

    def get(self):
        res = self.COST_TABLE[self.env.first_agent_perfs[self.idx] << self.first_shift | self.env.second_agent_perfs[
            self.idx] << self.second_shift]
        self.env.scores[self.second_shift] += res
        return res


class PrisonerDilemmaEnvironment(Environment):

    def __init__(self, first_agent, second_agent):
        self.first_agent, self.second_agent = first_agent, second_agent
        self.first_agent_perfs = []
        self.second_agent_perfs = []
        self.count = 0
        self.scores = [0, 0]

    def get_state(self, agent: Agent) -> State:
        state = np.array([[0.0, 0.0]])
        if self.count == 0:
            return state
        lst = self.first_agent_perfs if agent is self.second_agent else self.second_agent_perfs
        state[0][lst[-1]] = 1.0
        return state

    def perform(self, agent: Agent, action) -> Reward:
        _action = np.argmax(action)
        idx = self.count
        if agent is self.first_agent:
            self.first_agent_perfs.append(_action)
            first_shift = 1
            second_shift = 0
        else:
            first_shift = 0
            second_shift = 1
            self.second_agent_perfs.append(_action)
            self.count += 1
        return PrisonerDilemmaReward(self, idx, first_shift, second_shift), _action

    @property
    def should_continue(self):
        return self.count < 100

    @classmethod
    def game(cls, first_player, second_player, env=None, has_human_players=False):
        if env is None:
            env = cls(first_player, second_player)
        while True:
            if has_human_players:
                print(env)
            env.cycle(first_player)
            if not env.should_continue:
                break
            if has_human_players:
                print(env)
            env.cycle(second_player)
            if not env.should_continue:
                break
        first_player.conclude()
        second_player.conclude()
        if has_human_players:
            print(env)
        return env.scores

