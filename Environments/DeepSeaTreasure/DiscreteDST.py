SEED = 42


class DeepSeaTreasureEnvironment:
    grid_rows = 11
    grid_cols = 10

    depths = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
    treasure = [1, 34, 58, 78, 86, 92, 112, 116, 122, 124]

    # UP, DOWN, LEFT, RIGHT
    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    def __init__(self):
        self.reset()
        self.forbidden_states = self.__get_forbidden_states()
        self.treasure_locations = self.__get_treasure_locations()

    def __get_forbidden_states(self):
        forbidden_states = [(i, j) for j in range(self.grid_cols)
                            for i in range(self.depths[j] + 1, self.grid_rows)]
        return forbidden_states

    def __get_treasure_locations(self):
        treasure_locations = [(i, j) for j, i in enumerate(self.depths)]
        return treasure_locations

    def reset(self):
        self.n_steps = 0
        self.state = (0, 0)
        return self.state

    def step(self, action, episode=None):
        """
        Transition the environment through the input action
        """
        self.n_steps += 1
        # "Candidate" next location for the agent
        cand_loc = (self.state[0] + self.actions[action][0],
                    self.state[1] + self.actions[action][1])

        # Check if forbidden state
        if ((cand_loc[0] <= self.grid_rows - 1 and cand_loc[0] >= 0) and
                (cand_loc[1] <= self.grid_cols - 1 and cand_loc[1] >= 0) and
                (cand_loc not in self.forbidden_states)):
            # Set new state
            self.state = cand_loc

        rewards = self.get_rewards()
        state = self.state
        done = self.check_terminal_state()
        return state, rewards, done, None

    def get_rewards(self):
        rewards = [-1, 0]  # (time_penalty, treasure_reward)
        if self.state in self.treasure_locations:
            rewards[1] = self.treasure[self.state[1]]
        return tuple(rewards)

    def check_terminal_state(self):
        return (self.state in self.treasure_locations) or (self.n_steps > 200)
