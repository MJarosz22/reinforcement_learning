from QLearning import QLearning


class MyQLearning(QLearning):
    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        """
        :param state: current state
        :param action: action taken
        :param r: reward for taking the action
        :param state_next: the next state after performing action
        :param possible_actions: all possible actions in state_next
        :param alfa: learning rate
        :param gamma: discount
        """
        # Calculate the maximum Q-value for the next state
        max_q_next = max([self.get_q(state_next, a) for a in possible_actions])
        # Calculate the updated Q value
        q_updated = (1 - alfa) * self.get_q(state, action) + alfa * (r + gamma * max_q_next)
        # Update the Q table with the new Q value
        self.set_q(state, action, q_updated)
