import numpy as np
import random
import math
import matplotlib.pyplot as plt

class robot:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.reward = 0
        self.collection = 0

    def getCurrent(self, grid):
        return grid[self.x][self.y]

    def getNorth(self, grid):
        return grid[self.x - 1][self.y]

    def getSouth(self, grid):
        return grid[self.x + 1][self.y]

    def getEast(self, grid):
        return grid[self.x][self.y + 1]

    def getWest(self, grid):
        return grid[self.x][self.y - 1]

    def pickUp(self, grid):
        if grid[self.x][self.y] == 1:
            grid[self.x][self.y] = 0
            return True
        else:
            return False

    def moveNorth(self, grid):
        if self.getNorth(grid) == -1:
            return False
        self.x -= 1
        return True

    def moveSouth(self, grid):
        if (self.getSouth(grid) == -1):
            return False
        self.x += 1
        return True

    def moveEast(self, grid):
        if (self.getEast(grid) == -1):
            return False
        self.y += 1
        return True

    def moveWest(self, grid):
        if (self.getWest(grid) == -1):
            return False
        self.y -= 1
        return True

    def selectAction(self, curr_state, qTable, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, 4)
        poss_actions = [qTable[curr_state][i] for i in range(5)]
        max_q_value = max(poss_actions)
        max_indices = [i for i, q_value in enumerate(poss_actions) if q_value == max_q_value]
        action = np.random.choice(max_indices)
        return action

    def performAction(self, action, grid):
        actions = {
            0: {"reward": 10, "movement": self.pickUp},
            1: {"reward": 0, "movement": self.moveNorth},
            2: {"reward": 0, "movement": self.moveSouth},
            3: {"reward": 0, "movement": self.moveEast},
            4: {"reward": 0, "movement": self.moveWest}
        }
        reward = actions[action]["reward"]
        success = actions[action]["movement"](grid)
        if not success:
            reward = -5
        if action == 0 and success:
            self.collection += 1
        return reward
    
    def train_and_test(self, qTable):
        num_reps = 5000
        epsilon = 0.1
        rewards = []
        plot_rewards = []

        for i in range(num_reps):
            num_steps = 200
            eta = 0.2
            gamma = 0.9
            grid = np.random.randint(2, size=(12, 12))
            for j in range(len(grid)):
                for k in range(len(grid[j])):
                    if j == 0 or j == 11 or k == 0 or k == 11:
                        grid[j][k] = -1
            self.x, self.y = random.randint(1, 10), random.randint(1, 10)
            self.collection = 0
            self.reward = 0
            total_cans = np.sum(grid == 1)
            total_reward = 0
            for step in range(num_steps):
                curr_state = (self.getCurrent(grid), self.getNorth(grid), self.getSouth(grid), self.getEast(grid), self.getWest(grid))
                if curr_state not in qTable:
                    qTable[curr_state] = np.zeros(5)
                action = self.selectAction(curr_state, qTable, epsilon)
                reward = self.performAction(action, grid)
                total_reward += reward
                self.reward += reward
                new_state = (self.getCurrent(grid), self.getNorth(grid), self.getSouth(grid), self.getEast(grid), self.getWest(grid))
                if new_state not in qTable:
                    qTable[new_state] = np.zeros(5)
                qTable[curr_state][action] += eta * (reward + gamma * max(qTable[new_state]) - qTable[curr_state][action])
                if reward == 10:
                    self.collection += 1
                    grid[self.y][self.x] = 0
                    self.x, self.y = random.randint(1, 10), random.randint(1, 10)
                    while grid[self.y][self.x] == -1:
                        self.x, self.y = random.randint(1, 10), random.randint(1, 10)

            lost = (self.collection * 10) - self.reward
            print("Episodes:", str(i), "   Total Reward:", str(self.reward), "   Points Lost:", str(lost), "   Cans Collected:", str(self.collection))
            if i % 50 == 0:
                epsilon -= 0.001
            if i % 100 == 0:
                plot_rewards.append(self.reward)
            rewards.append(self.reward)
        mean_reward = sum(rewards) / num_reps
        std_dev = math.sqrt(sum((x - mean_reward) ** 2 for x in rewards) / num_reps)
        print("Average Reward: ", mean_reward)
        print("Standard Deviation: ", std_dev)
        plt.plot(plot_rewards)
        plt.show()