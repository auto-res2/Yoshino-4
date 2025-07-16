import traceback
import time
import random

tasks = {
    "simple_stats": {"complexity": 1},
    "linear_regression": {"complexity": 2},
    "neural_pipeline": {"complexity": 3}
}

class TreeNode:
    def __init__(self, task_name, code, metric, iterations):
        self.task_name = task_name
        self.code = code
        self.metric = metric
        self.iterations = iterations
        self.children = []

def safe_exec(code_str, extra_locals=None):
    """
    Safely execute a code string using exec.
    Captures errors and returns the traceback if any.
    The provided extra_locals can include necessary functions.
    """
    if extra_locals is None:
        extra_locals = {}
    try:
        exec(code_str, {}, extra_locals)
        return None
    except Exception as e:
        return traceback.format_exc()

def debug_loop(code_str, max_iterations=3):
    """
    Iteratively execute the provided code string.
    If an error is caught (here, a size mismatch),
    then simulate a debugging step by replacing the wrong dimension.
    """
    for iteration in range(max_iterations):
        local_context = {}
        err = safe_exec(code_str, local_context)
        if err is None:
            print(f"Code executed successfully on iteration {iteration}.")
            return code_str, iteration
        else:
            print(f"Error captured on iteration {iteration}:")
            print(err)
            if ("size mismatch" in err or "in_features" in err):
                print("Detected size mismatch error. Attempting to fix nn.Linear dimension...")
                code_str = code_str.replace("nn.Linear(2, 1)", "nn.Linear(1, 1)")
            else:
                code_str = code_str.replace("optimizer.step()", "optimizer.step()")
    print("Max debugging iterations reached.")
    return code_str, max_iterations

def evaluate_task(task_name, strategy):
    """
    Simulate task evaluation.
    One-shot strategy is fast but may yield poorer performance for complex tasks.
    Iterative debugging is slower but yields higher performance.
    Returns (iterations, runtime in seconds, performance metric).
    """
    start_time = time.time()
    if strategy == 'one-shot':
        iterations = 1
        if tasks[task_name]["complexity"] == 1:
            performance_metric = random.uniform(0.9, 1.0)
        else:
            performance_metric = random.uniform(0.5, 0.7)
    else:
        iterations = random.randint(2, 4)
        performance_metric = random.uniform(0.85, 1.0)
    runtime = time.time() - start_time
    return iterations, runtime, performance_metric

def adaptive_strategy(task_name):
    """
    Choose between one-shot or iterative strategy based on task complexity.
    Build a TreeNode with candidate solution data.
    """
    if tasks[task_name]["complexity"] <= 1:
        chosen_strategy = 'one-shot'
    else:
        chosen_strategy = 'iterative'
        
    print(f"Task '{task_name}': Using {chosen_strategy} strategy.")
    iterations, runtime, metric = evaluate_task(task_name, chosen_strategy)
    code_placeholder = f"# {task_name} code using {chosen_strategy} strategy"
    node = TreeNode(task_name, code_placeholder, metric, iterations)
    print(f"Task: {task_name}, Iterations: {iterations}, Runtime: {runtime:.4f}s, Performance Metric: {metric:.2f}")
    return node
