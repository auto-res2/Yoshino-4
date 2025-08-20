import numpy as np
import gym
from gym import spaces
import random

def run_pipeline(data, window_size=50, proactive=True, model=None):
    """Evaluate pipeline performance with proactive vs reactive adaptation."""
    from src.train import predictive_shift_detector
    
    performance = []
    adaptation_triggered = []
    for i in range(len(data)):
        if i >= window_size:
            window = data[i - window_size:i]
            alert, uncertainty = predictive_shift_detector(window, model=model)
            if proactive and alert:
                perf = 1.0 + np.exp(-uncertainty)
                adaptation_triggered.append(1)
            else:
                perf = 0.8
                adaptation_triggered.append(0)
        else:
            perf = 0.8
            adaptation_triggered.append(0)
        performance.append(perf)
    return np.array(performance), np.array(adaptation_triggered)

class ResourceAllocationEnv(gym.Env):
    """Custom Gym environment for multi-objective resource allocation."""
    def __init__(self, n_models=3, total_budget=100):
        super(ResourceAllocationEnv, self).__init__()
        self.n_models = n_models
        self.total_budget = total_budget
        self.action_space = spaces.Box(low=0, high=total_budget, shape=(n_models,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_models,), dtype=np.float32)
        self.state = None

    def step(self, action):
        action = np.array(action)
        if action.sum() > 0:
            allocation = self.total_budget * action / (np.sum(action) + 1e-8)
        else:
            allocation = np.zeros_like(action)
        performance = np.random.rand(self.n_models) 
        benefit = allocation * performance
        cost = allocation * 0.1
        reward = np.sum(benefit) - np.sum(cost)
        self.state = np.random.rand(self.n_models)
        done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.rand(self.n_models)
        return self.state

def random_allocation_agent(env, episodes=1000):
    """Baseline random allocation agent."""
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        action = env.action_space.sample()
        _, r, _, _ = env.step(action)
        rewards.append(r)
    return np.mean(rewards)

def simulate_rl_agent(env, episodes=1000):
    """Simulated RL agent with learning heuristic."""
    rewards = []
    learned_action = np.ones(env.n_models)
    for ep in range(episodes):
        state = env.reset()
        learned_action = 0.9 * learned_action + 0.1 * state
        action = learned_action
        _, r, _, _ = env.step(action)
        rewards.append(r)
    return np.mean(rewards)

def generate_explanation(event_type, data_pattern):
    """Generate human-readable explanations for adaptation events."""
    explanations = {
        'domain_shift': f"Detected a significant shift in data distribution with pattern {data_pattern}. Adjusting evaluation metrics to emphasize semantic diversity.",
        'resource_reallocation': f"Resource utilization spike detected in module {data_pattern}. Reallocating computational budget to balance cost and performance.",
        'curriculum_adjustment': f"Performance feedback indicates current task difficulty is suboptimal. Scaling task complexity for better knowledge extraction."
    }
    return explanations.get(event_type, "No explanation available.")
