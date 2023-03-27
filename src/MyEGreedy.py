import numpy as np

class MyEGreedy:
    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        """
        Selects a random action for the agent in some state in the maze
        :param agent
        :param maze
        :return: random action
        """
        actions = maze.get_valid_actions(agent)
        return np.random.choice(actions)

    def get_best_action(self, agent, maze, q_learning):
        """
        Selects the best possible action for the agent in some state in the maze
        :param agent
        :param maze
        :param q_learning
        :return: best action
        """
        s = agent.get_state(maze)
        actions = maze.get_valid_actions(agent)
        values = q_learning.get_action_values(s, actions)
        max_rewards = np.where(values == np.max(values))[0]
        # If there are multiple best actions, select a random one
        chosen_index = np.random.choice(max_rewards)
        return actions[chosen_index]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        """
        Selects the best action with probability (1 - epsilon) for the agent;
        random action with probability (epsilon)
        :param agent:
        :param maze:
        :param q_learning:
        :param epsilon:
        :return:
        """
        rand = np.random.uniform()
        if rand < epsilon:
            return self.get_random_action(agent, maze)
        else:
            return self.get_best_action(agent, maze, q_learning)
