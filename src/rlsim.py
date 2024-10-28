import pygame
import numpy as np
import random
from collections import defaultdict
import pickle
import os

class Environment:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.cell_size = 20  # Size of each grid cell
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RL Agent Simulation")
        
        # Initialize font
        self.font = pygame.font.Font(None, 24)  # Default font, size 24
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        
        # Stats tracking
        self.current_episode = 0
        self.current_step = 0
        self.current_reward = 0
        
        self.reset()

    def reset(self):
        # Reset step counter
        self.current_step = 0
        
        # Reset agent position to random location
        self.agent_pos = [
            random.randint(0, (self.width//self.cell_size)-1) * self.cell_size,
            random.randint(0, (self.height//self.cell_size)-1) * self.cell_size
        ]
        
        # Reset ball position to random location
        self.ball_pos = [
            random.randint(0, (self.width//self.cell_size)-1) * self.cell_size,
            random.randint(0, (self.height//self.cell_size)-1) * self.cell_size
        ]
        
        # Set target zone position (fixed)
        self.target_zone = [
            self.width - 2 * self.cell_size,
            self.height - 2 * self.cell_size,
            self.cell_size * 2,
            self.cell_size * 2
        ]
        
        self.has_ball = False
        return self._get_state()

    def _get_state(self):
        # State consists of: agent_x, agent_y, ball_x, ball_y, has_ball
        if self.has_ball:
            ball_pos = self.agent_pos
        else:
            ball_pos = self.ball_pos
            
        return (
            self.agent_pos[0] // self.cell_size,
            self.agent_pos[1] // self.cell_size,
            ball_pos[0] // self.cell_size,
            ball_pos[1] // self.cell_size,
            int(self.has_ball)
        )

    def step(self, action):
        self.current_step += 1
        # Actions: 0: up, 1: right, 2: down, 3: left
        moves = [
            (0, -self.cell_size),  # up
            (self.cell_size, 0),   # right
            (0, self.cell_size),   # down
            (-self.cell_size, 0)   # left
        ]
        
        # Move agent
        dx, dy = moves[action]
        new_x = max(0, min(self.width - self.cell_size, self.agent_pos[0] + dx))
        new_y = max(0, min(self.height - self.cell_size, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]
        
        # Check for ball pickup
        reward = -1  # Small negative reward for each step
        done = False
        
        if not self.has_ball:
            if self.agent_pos == self.ball_pos:
                self.has_ball = True
                reward += 75  # Reward for picking up ball
        
        # Check if agent with ball is in target zone
        if self.has_ball:
            if (self.target_zone[0] <= self.agent_pos[0] <= self.target_zone[0] + self.target_zone[2] and
                self.target_zone[1] <= self.agent_pos[1] <= self.target_zone[1] + self.target_zone[3]):
                reward += 150  # Reward for delivering ball
                done = True
        
        return self._get_state(), reward, done

    def render(self, episode, total_reward, max_steps):
        self.screen.fill(self.WHITE)
        
        # Draw target zone
        pygame.draw.rect(self.screen, self.GREEN, self.target_zone)
        
        # Draw agent
        pygame.draw.rect(self.screen, self.RED, 
                        (self.agent_pos[0], self.agent_pos[1], 
                         self.cell_size, self.cell_size))
        
        # Draw ball if not picked up
        if not self.has_ball:
            pygame.draw.circle(self.screen, self.BLUE,
                             (self.ball_pos[0] + self.cell_size//2,
                              self.ball_pos[1] + self.cell_size//2),
                             self.cell_size//2)
        
        # Render stats
        stats_texts = [
            f"Episode: {episode}",
            f"Steps Left: {max_steps - self.current_step}",
            f"Reward: {total_reward:.1f}"
        ]
        
        for i, text in enumerate(stats_texts):
            text_surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()

class QLearningAgent:
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space_size = action_space_size
        
    def save(self, filename):
        # Convert defaultdict to regular dict for saving
        q_dict = dict(self.q_table)
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': q_dict,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'action_space_size': self.action_space_size
            }, f)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        agent = cls(
            action_space_size=data['action_space_size'],
            learning_rate=data['lr'],
            discount_factor=data['gamma'],
            epsilon=data['epsilon']
        )
        
        # Convert saved dict back to defaultdict
        agent.q_table = defaultdict(lambda: np.zeros(data['action_space_size']),
                                  data['q_table'])

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

def train():
    env = Environment()
    
    # Try to load existing agent, or create new one
    save_file = "agent.sav"
    if os.path.exists(save_file):
        print("Loading existing agent from", save_file)
        try:
            agent = QLearningAgent.load(save_file)
        except Exception as e:
            print("Error loading agent:", e)
            print("Creating new agent instead")
            agent = QLearningAgent(action_space_size=4)
    else:
        print("Creating new agent")
        agent = QLearningAgent(action_space_size=4)
    
    episodes = 1000
    max_steps = 200
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            
            env.render(episode, total_reward, max_steps)
            pygame.time.wait(30)  # Slow down visualization
            
            if done:
                break
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            # Save agent's progress
            try:
                agent.save("agent.sav")
                print(f"Saved agent progress to agent.sav")
            except Exception as e:
                print("Error saving agent:", e)
            
    pygame.quit()

if __name__ == "__main__":
    train()
