import time
import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
import time
import sys
import argparse
from instance_generator import generate_instances



def assign_min_sum(cost_matrix):
    """
    Solve the minimum sum assignment problem using the Hungarian algorithm.
    
    :param cost_matrix: Numpy array of size (n, m) containing the costs
    :return: Tuple containing the optimal assignment and its corresponding value
    """
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    return col_ind, cost_matrix[row_ind, col_ind].sum()


def owa(y, w):
    """
    Compute the ordered weighted average (OWA) of a vector y with weights w.
    
    :param y: Numpy array of size n containing the values
    :param w: Numpy array of size n containing the weights
    :return: The OWA value
    """
    sorted_y = np.sort(y)[::-1]  # Sort y in descending order
    return float(np.dot(sorted_y, w))


def build_pos(n, delta):
    """
    Build the POS array based on the number of costs n and delta value.
    
    :param n: Number of costs to be considered in the optimization problem
    :param delta: Delta value for the construction of the grid of possible values for t
    :return: Ordered list containing the POS values
    """
    pos, s = set(), 0
    while True:
        r = int(np.ceil((1 + delta) ** s - 1e-10)) - 1
        if r >= n - 1:
            break
        pos.add(r)
        s += 1
    pos.add(n - 1)
    return sorted(pos)




def explore(l, prev_fixed, POS, eps, n, w, cost_matrix, min_sum_func, state, max_j):
    """
    Recursive function to explore the possible values of t and find the optimal solution (from Chakrabarty and Swamy).
    
    :param l: Rank of the current element being chosen in t (must be greater than 0)
    :param prev_fixed: Numpy array of size p containing the previous fixed elements (with last elements being 0)
    :param p: Size of vector t (number of elements to choose)
    :param eps: Epsilon value for the construction of the grid of possible values for t
    :param n: Number of costs to be considered in the optimization problem
    """
    p = len(POS)

    if l < 1:
        raise ValueError("l must be greater than 0")
    
    if l == p:
        t = prev_fixed.copy()

        # Check if the current t is a valid solution (decreasing, non-negative, and of size p)
        if np.all(t[:-1] >= t[1:]) and np.all(t >= 0) and np.shape(t)[0] == p:

            # Modified costs based on t
            costs_t = np.zeros(cost_matrix.shape, dtype=float)
            for k in range(p-1):
                costs_t += (w[POS[k]] - w[POS[k+1]]) * np.maximum(cost_matrix - t[k], 0)

            costs_t += w[POS[p-1]] * np.maximum(cost_matrix - t[p-1], 0) 
            
            # Solve the optimization problem with the modified costs
            sol, _ = min_sum_func(costs_t)
            # Compute the OWA value for the solution
            value = owa(cost_matrix[np.arange(len(sol)), sol], w)
            state["num_t"] += 1

            if state["best_value"] is None or value < state["best_value"]:
                state["best_sol"] = sol.copy()
                state["best_value"] = value

            return 

        else:
            raise ValueError("t is not a valid solution")
        

    o1 = prev_fixed[0]
    last_fixed = prev_fixed[l-1]

    candidates = [o1 / (1 + eps) ** j for j in range(max_j + 1) if o1 / (1 + eps) ** j <= last_fixed] 
    candidates.append(0.0)

    for candidate in candidates:
        prev_fixed[l] = candidate
        explore(l+1, prev_fixed, POS, eps, n, w, cost_matrix, min_sum_func, state, max_j)




def approx_ordered(cost_matrix, w, delta, eps, min_sum_func=assign_min_sum):
    """
    Approximate ordered optimization problem solver.
    
    :param cost_matrix: Numpy array of size (n, m) containing the costs
    :param w: Numpy array of size n containing the weights
    :param delta: Delta value for the construction of the grid of possible values for t
    :param eps: Epsilon value for the construction of the grid of possible values for t
    :return: Tuple containing the best solution found and its corresponding value
    """

    n = len(w)

    # Compute POS and p

    POS = build_pos(n, delta)
    p = len(POS)

    state = {"best_sol": None, "best_value": None, "num_t": 0}

    # Case o1 = 0
    sol, val = min_sum_func((cost_matrix > 0).astype(float))
    if val == 0:
        return sol, val, POS, 1

    # Case o1 > 0
    c_min, c_max = np.min(cost_matrix > 0), np.max(cost_matrix)
    k_min = int(np.floor(np.log(c_min) / np.log(1 + eps)))
    k_max = int(np.ceil(np.log(c_max) / np.log(1 + eps)))
    S = [(1 + eps) ** k for k in range(k_min, k_max + 1)]  
    max_j = int(np.floor(np.log((1 + eps) * n / eps) / np.log(1 + eps)))

    for rho in S:
        if rho == 0:
            continue

        prev_fixed = np.zeros(p)
        prev_fixed[0] = rho

        explore(1, prev_fixed, POS, eps, n, w, cost_matrix, min_sum_func, state, max_j)

    return state["best_sol"], state["best_value"], POS, state["num_t"]



# ---------------------------------------------------------------------------
# Experiments: one row per instance, appended to the CSV as soon as it is done
# ---------------------------------------------------------------------------
 
def results_path(results_dir, n, delta, eps, w_type):
    """One file per configuration, so runs with different parameters never overwrite each other."""
    return os.path.join(results_dir, f"approx_n{n}_delta{delta}_eps{eps}_{w_type}.csv")
 

 
def test_approx_assign(n, delta, eps, w_type, nb_instances, save=True, seed=0,
                       results_dir="../results/assignment", save_solutions=True):
    path = results_path(results_dir, n, delta, eps, w_type)
    done = set()
    if save:
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(path):                                   # resume an interrupted run
            done = set(pd.read_csv(path)["instance"].astype(str))
            print(f"Resuming: {len(done)} instance(s) already in {path}")
 
    instances = generate_instances(n, nb_instances, w_type, cost_seed=seed)
    rows = []
 
    for i, instance in enumerate(instances):
        if str(instance["instance"]) in done:
            continue
 
        t0 = time.perf_counter()
        best_sol, best_value, POS, num_t = approx_ordered(instance["cost_matrix"], instance["W"],
                                                          delta=delta, eps=eps)
        elapsed = time.perf_counter() - t0
 
        row = {
            "method": "chen_appendixC",
            "n": n, "delta": delta, "eps": eps, "w_type": w_type, "seed": seed,
            "instance": instance["instance"],
            "best_value": best_value,
            "num_t": num_t,
            "p": len(POS),
            "time_s": elapsed,
            "guarantee": (1 + delta) * (1 + 2 * eps),
        }
        if save_solutions:
            row["assignment"] = " ".join(map(str, best_sol))       # agent i -> task assignment[i]
        rows.append(row)
 
        if save:
            pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))
            print(f"[chen_appendixC] n={n}, instance {i + 1}/{nb_instances} saved "
                  f"(value={best_value:.2f}, {num_t} vectors, {elapsed:.1f}s)", flush=True)
 
    return pd.read_csv(path) if save else pd.DataFrame(rows)
 
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=10)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--w-type", default="s_gini_1.5_weights")
    parser.add_argument("--nb-instances", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=3600.0,
                         help="mean budget (s) per (method, n) -- default 3600s.")

    args = parser.parse_args()

    n, delta, eps, w_type = args.n, args.delta, args.eps, args.w_type

    test_approx_assign(n=n, delta=delta, eps=eps, w_type=w_type, nb_instances=30, save=True, seed=0)
 
