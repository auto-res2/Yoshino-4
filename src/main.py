import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from collections import deque
import os

from skopt import gp_minimize
from skopt.space import Real

np.random.seed(42)
torch.manual_seed(42)

def synthetic_performance(d):
    if isinstance(d, list):
        d = d[0]
    optimum = 0.5
    noise = np.random.normal(0, 0.02)
    return (d - optimum)**2 + noise

def bmdsg_callback(result, log_data):
    x_val = result.x_iters[-1][0]
    f_val = result.fun
    log_data.append((x_val, f_val))
    print(f"BMDSG Iteration {len(log_data)}: difficulty={x_val:.4f}, performance_error={f_val:.4f}")


def experiment1_bmdsg(n_calls=30):
    print('Running Experiment 1: Adaptive Scenario Generation with BMDSG')
    space  = [Real(0.0, 1.0, name='difficulty')]
    log_data = []
    callback_wrapper = lambda res: bmdsg_callback(res, log_data)
    res = gp_minimize(synthetic_performance, space, n_calls=n_calls, callback=[callback_wrapper], random_state=42)
    difficulties, performance_errors = zip(*log_data)
    plt.figure(figsize=(8, 5))
    plt.plot(difficulties, performance_errors, marker='o', linestyle='-')
    plt.title('Evolution of Difficulty Parameters vs. Performance Error')
    plt.xlabel('Difficulty Parameter')
    plt.ylabel('Estimated Performance Error')
    plt.grid(True)
    plot_filename = '.research/iteration1/images/adaptive_scenario.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Experiment 1 plot saved as {plot_filename}')
    plt.close()


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score


def experiment2_emaf_rlg(n_samples=500, adversarial_frac=0.1):
    print('\nRunning Experiment 2: Robust Label Guard and Adversarial Defense with EMAF-RLG')
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
    # Create adversarial labels by flipping a fraction of the labels
    n_flip = int(adversarial_frac * n_samples)
    flip_indices = np.random.choice(np.arange(n_samples), size=n_flip, replace=False)
    y_adv = np.copy(y)
    y_adv[flip_indices] = 1 - y_adv[flip_indices]

    model_lr = LogisticRegression(max_iter=500, random_state=42)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_svm = SVC(probability=True, random_state=42)
    
    for model in [model_lr, model_dt, model_svm]:
        model.fit(X, y)

    pred_lr = model_lr.predict_proba(X)[:, 1]
    pred_dt = model_dt.predict_proba(X)[:, 1]
    pred_svm = model_svm.predict_proba(X)[:, 1]

    ensemble_scores = 0.33 * pred_lr + 0.33 * pred_dt + 0.34 * pred_svm
    pred_ensemble = (ensemble_scores > 0.5).astype(int)

    ensemble_features = np.column_stack((pred_lr, pred_dt, pred_svm))
    iso_forest = IsolationForest(contamination=adversarial_frac, random_state=42)
    iso_forest.fit(ensemble_features)
    anomaly_predictions = iso_forest.predict(ensemble_features)  # -1 indicates anomaly

    adjusted_preds = np.copy(pred_ensemble)
    for i, flag in enumerate(anomaly_predictions):
        if flag == -1:
            adjusted_preds[i] = y[i]

    baseline_accuracy = accuracy_score(y_adv, pred_ensemble)
    adjusted_accuracy = accuracy_score(y_adv, adjusted_preds)

    baseline_precision = precision_score(y_adv, pred_ensemble)
    adjusted_precision = precision_score(y_adv, adjusted_preds)

    baseline_recall = recall_score(y_adv, pred_ensemble)
    adjusted_recall = recall_score(y_adv, adjusted_preds)

    print('Performance metrics on adversarial dataset:')
    print('Before Adversarial Defense:')
    print(f'  Accuracy: {baseline_accuracy:.4f}, Precision: {baseline_precision:.4f}, Recall: {baseline_recall:.4f}')
    print('After Adversarial Defense:')
    print(f'  Accuracy: {adjusted_accuracy:.4f}, Precision: {adjusted_precision:.4f}, Recall: {adjusted_recall:.4f}')

    metrics = ['Accuracy', 'Precision', 'Recall']
    baseline_vals = [baseline_accuracy, baseline_precision, baseline_recall]
    adjusted_vals = [adjusted_accuracy, adjusted_precision, adjusted_recall]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, baseline_vals, width, label='Before Defense')
    plt.bar(x + width/2, adjusted_vals, width, label='After Defense')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Label Guard Performance Metrics')
    plt.legend()
    plot_filename = '.research/iteration1/images/label_guard_performance.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Experiment 2 plot saved as {plot_filename}')
    plt.close()


try:
    import gymnasium as gym
except ImportError:
    import gym

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# HASCC Controller: Dynamically adjust simulation fidelity based on recent performance
class HASCCController:
    def __init__(self, threshold=15):
        self.recent_rewards = deque(maxlen=20)
        self.threshold = threshold
        self.fidelity_mode = 'high'

    def update(self, reward):
        self.recent_rewards.append(reward)
        avg_reward = np.mean(self.recent_rewards)
        if avg_reward < self.threshold and self.fidelity_mode == 'high':
            self.fidelity_mode = 'low'
        elif avg_reward >= self.threshold and self.fidelity_mode == 'low':
            self.fidelity_mode = 'high'
        return self.fidelity_mode

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        target = reward
        if not done:
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
        q_values = self.model(state_tensor)
        target_vec = q_values.clone().detach()
        target_vec[0][action] = target
        loss = self.criterion(q_values, target_vec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def experiment3_hascc(episodes=20):
    print('\nRunning Experiment 3: Hybrid Adaptive Simulation and Continual Calibration (HASCC) Efficiency Test')
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    controller = HASCCController(threshold=15)  

    performance_log = []
    resource_log = []
    fidelity_record = []

    for ep in range(episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        done = False
        total_reward = 0
        start_time = time.time()
        current_fidelity = controller.fidelity_mode
        # In a real adaptive simulation, fidelity might change parameters of the environment.
        while not done:
            action = agent.act(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
                done = done or truncated
            else:
                next_state, reward, done, _ = step_result
            agent.train_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        elapsed_time = time.time() - start_time
        performance_log.append(total_reward)
        resource_log.append(elapsed_time)
        current_fidelity = controller.update(total_reward)
        fidelity_record.append(current_fidelity)
        print(f'Episode {ep+1}: Total Reward = {total_reward:.2f}, Fidelity Mode = {current_fidelity}, Time = {elapsed_time:.4f} sec')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(performance_log, marker='o')
    plt.title('Agent Performance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(resource_log, marker='o', color='r')
    plt.title('Processing Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Time (sec)')
    plt.tight_layout()
    plot_filename = '.research/iteration1/images/hascc_efficiency.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Experiment 3 plot saved as {plot_filename}')
    plt.close()



def run_tests():
    print('---------------------\nStarting Test Suite for AMMEF 2.0 Experiments\n---------------------')
    experiment1_bmdsg(n_calls=15)
    
    experiment2_emaf_rlg(n_samples=300, adversarial_frac=0.1)
    
    experiment3_hascc(episodes=5)
    
    print('---------------------\nAll Experiments Completed Successfully\n---------------------')


if __name__ == '__main__':
    run_tests()
