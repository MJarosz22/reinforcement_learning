from src.Agent import Agent
from src.Maze import Maze
from src.MyEGreedy import MyEGreedy
from src.MyQLearning import MyQLearning
from matplotlib import pyplot as plt


def run_optimization():
    # Load the maze
    file = "../data/toy_maze.txt"
    maze = Maze(file)

    # Set the reward at the bottom right to 10
    maze.set_reward(maze.get_state(9, 9), 10)
    maze.set_reward(maze.get_state(9, 0), 5)

    # Create a robot at starting and reset location (0,0) (top left)
    robot = Agent(0, 0)

    # Make a selection object (you need to implement the methods in this class)
    selection = MyEGreedy()

    # Make a Qlearning object (you need to implement the methods in this class)
    learn = MyQLearning()

    # Hyper parameters
    epsilon = 0.1
    alfa = 0.7
    gamma = 0.9
    episodes = 10
    steps = 30000

    # Set the starting parameters
    stop = False
    count = 0
    ep = 0

    steps_done = []
    # Keep learning for the number of episodes/steps
    while not stop:
        count += 1
        state = robot.get_state(maze)
        # Select action
        action = selection.get_egreedy_action(robot, maze, learn, epsilon)
        # Make a move
        state_next = robot.do_action(action, maze)
        r = maze.get_reward(state_next)
        possible_actions = maze.get_valid_actions(robot)
        # Update Q-table
        learn.update_q(state, action, r, state_next, possible_actions, alfa, gamma)

        # Reset if robot reached the goal
        if robot.get_state(maze) == maze.get_state(9, 9) or robot.get_state(maze) == maze.get_state(9, 0):
            ep += 1
            steps_done.append(robot.nr_of_actions_since_reset)
            robot.reset()
        # Stopping criterion
        if count == steps or ep == episodes:
            stop = True

    x = range(0, episodes)
    plt.plot(x, steps_done)
    plt.xlabel("Current episode")
    plt.ylabel("Steps taken til goal reached")
    plt.show()
