import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import scipy.optimize as opt
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from instance_generator import generate_instances
from resume import RERUN_MODES, append_row, prepare_resume

from itertools import combinations_with_replacement


class TimeLimitReached(Exception):
    pass


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




def explore(l, prev_fixed, POS, w, cost_matrix, min_sum_func, state, min_j, candidates, buffers, coefs, tmp):
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
        # Check if the time limit has been reached
        if state["deadline"] is not None and time.monotonic() >= state["deadline"]:
            raise TimeLimitReached
        '''
        #t = prev_fixed.copy()
        t = prev_fixed

        # Check if the current t is a valid solution (decreasing, non-negative, and of size p)
        #if np.all(t[:-1] >= t[1:]) and np.all(t >= 0) and np.shape(t)[0] == p:

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

        #else:
        #raise ValueError("t is not a valid solution")
        
    t1 = prev_fixed[0]
    last_fixed = prev_fixed[l-1]
    min_j = np.ceil(np.log(t1 / last_fixed) / np.log(1 + eps) - 1e-9) # if last_fixed = 0, then the algorithm will not explore any further, as all remaining candidates will be 0.0
    # 1e-9 is added to avoid floating point issues when t1 / last_fixed is very close to a power of (1 + eps)
 
    #candidates.append(0.0)

    for candidate in candidates[min_j:]:
        prev_fixed[l] = candidate
        sum_costs_t += (w[POS[l]] - w[POS[l+1]]) * np.maximum(cost_matrix - candidate, 0)
        explore(l+1, prev_fixed, POS, eps, n, w, cost_matrix, min_sum_func, state, min_j, candidates, sum_costs_t)

    # if candidate == 0.0, skip the remaining recursive calls, as they will all be 0.0 and will not yield a better solution
    prev_fixed[l:] = 0.0 * (p - l)  # Set the remaining elements to 0
    sum_costs_t += (w[POS[l]] - w[POS[p-1]]) * np.maximum(cost_matrix - candidate, 0)
    explore(p, prev_fixed, POS, eps, n, w, cost_matrix, min_sum_func, state, min_j, candidates, sum_costs_t)
    '''
        # Solve the optimization problem with the modified costs
        sol, _ = min_sum_func(buffers[p]) # buffers[p] contains the sum of costs on t
        # Compute the OWA value for the solution
        value = owa(cost_matrix[np.arange(len(sol)), sol], w)
        state["num_t"] += 1

        if state["best_value"] is None or value < state["best_value"]:
            state["best_sol"] = sol.copy()
            state["best_value"] = value

        return 
    
    for j in range(min_j, len(candidates)):
        c = candidates[j]
        prev_fixed[l] = c
        np.subtract(cost_matrix, c, out=tmp)
        np.maximum(tmp, 0.0, out=tmp)
        tmp *= coefs[l] # multiply matrix costs_t by (w_l - w_next(l))
        np.add(buffers[l], tmp, out=buffers[l+1])  # add costs_t to the sum already computed in buffers[l] and put it in buffers[l+1]
        explore(l+1, prev_fixed, POS, w, cost_matrix, min_sum_func, state, j, candidates, buffers, coefs, tmp)





def approx_ordered(cost_matrix, w, delta, eps, min_sum_func=assign_min_sum, deadline=None):
    """
    Approximate ordered optimization problem solver based on the algorithm from Chakrabarty and Swamy (2019).
    
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

    state = {"best_sol": None, "best_value": None, "num_t": 0, "deadline": deadline}

    # Case rho = 0
    sol, val = min_sum_func((cost_matrix > 0).astype(float))
    if val == 0:
        return sol, val, POS, 1, "completed"

    # Case rho > 0
    c_min, c_max = cost_matrix[cost_matrix > 0].min(), cost_matrix.max()
    k_min = int(np.floor(np.log(c_min) / np.log(1 + eps)))
    k_max = int(np.ceil(np.log(c_max) / np.log(1 + eps)))
    S = [(1 + eps) ** k for k in range(k_min, k_max + 1)]  
    max_j = int(np.floor(np.log((1 + eps) * n / eps) / np.log(1 + eps)))
    min_j = 0

    wp = np.asarray(w, dtype=float)[POS] # keep only weights at k in POS
    coefs = np.append(wp[:-1] - wp[1:], wp[-1])
    buffers = np.zeros((p+1,) + cost_matrix.shape)  # buffers[0] = 0; keeps track of (w_l - w_next(l))(v-t_l) for each value of t and each value v in cost matrix
    base_cost = 1 / (1 + eps) ** np.arange(max_j + 1) # possible base values of threshold vector (to be multiplied by rho)
    tmp = np.empty(cost_matrix.shape)

    status = "completed"
    try:
        for rho in S:
            if rho == 0:
                continue

            prev_fixed = np.zeros(p)
            prev_fixed[0] = rho

            #candidates = [rho / (1 + eps) ** j for j in range(max_j + 1)] 
            candidates = rho * base_cost
            np.subtract(cost_matrix, rho, out=tmp)
            np.maximum(tmp, 0.0, out=tmp)
            np.multiply(tmp, coefs[0], out=buffers[1])

            explore(1, prev_fixed, POS, w, cost_matrix, min_sum_func, state, min_j, candidates, buffers, coefs, tmp)

    except TimeLimitReached:
        status = "timeout"

    return state["best_sol"], state["best_value"], POS, state["num_t"], status




def approx_ordered_loop(cost_matrix, w, delta, eps, min_sum_func=assign_min_sum, deadline=None):
    n = len(w)
    POS = build_pos(n, delta)
    p = len(POS)

    state = {"best_sol": None, "best_value": None, "num_t": 0, "deadline": deadline}

    # Case rho = 0
    sol, val = min_sum_func((cost_matrix > 0).astype(float))
    if val == 0:
        return sol, val, POS, 1, "completed"

    # Case rho > 0
    c_min, c_max = cost_matrix[cost_matrix > 0].min(), cost_matrix.max()
    k_min = int(np.floor(np.log(c_min) / np.log(1 + eps)))
    k_max = int(np.ceil(np.log(c_max) / np.log(1 + eps)))
    S = [(1 + eps) ** k for k in range(k_min, k_max + 1)]
    max_j = int(np.floor(np.log((1 + eps) * n / eps) / np.log(1 + eps)))

    wp = np.asarray(w, dtype=float)[POS]
    coefs = np.append(wp[:-1] - wp[1:], wp[-1])        # coefficient of term k
    buffers = np.zeros((p + 1,) + cost_matrix.shape)   # buffers[l] = sum of terms 0..l-1
    base_cost = np.append((1 + eps) ** -np.arange(max_j + 1), 0.0)  # 0 last
    idx = range(len(base_cost))
    rows = np.arange(cost_matrix.shape[0])

    status = "completed"
    try:
        for rho in S:
            candidates = base_cost * rho

            # Term 0: t[0] = rho
            np.subtract(cost_matrix, rho, out=buffers[1])
            np.maximum(buffers[1], 0.0, out=buffers[1])
            buffers[1] *= coefs[0]

            prev = None
            for s in combinations_with_replacement(idx, p - 1):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeLimitReached

                # First position that differs from the previous tuple
                start = 0
                if prev is not None:
                    while s[start] == prev[start]:
                        start += 1

                # Only recompute the terms after the shared prefix
                for i in range(start, p - 1):
                    out = buffers[i + 2]
                    np.subtract(cost_matrix, candidates[s[i]], out=out)
                    np.maximum(out, 0.0, out=out)
                    out *= coefs[i + 1]
                    out += buffers[i + 1]

                sol, _ = min_sum_func(buffers[p])
                value = owa(cost_matrix[rows, sol], w)
                state["num_t"] += 1
                if state["best_value"] is None or value < state["best_value"]:
                    state["best_sol"] = sol.copy()
                    state["best_value"] = value

                prev = s

    except TimeLimitReached:
        status = "timeout"

    return state["best_sol"], state["best_value"], POS, state["num_t"], status
 
                



# ---------------------------------------------------------------------------
# Experiments: one row per instance, appended to the CSV as soon as it is done
# ---------------------------------------------------------------------------
 
def results_path(results_dir, n, delta, eps, w_type, loop=False):
    """One file per configuration, so runs with different parameters never overwrite each other."""
    if loop:
        return os.path.join(results_dir, f"approx_n{n}_delta{delta}_eps{eps}_{w_type}_loop.csv")
    return os.path.join(results_dir, f"approx_n{n}_delta{delta}_eps{eps}_{w_type}.csv")
    i


def n_available_cores():
    """Cores actually allocated (cpuset OAR), not all cores of the node."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # hors Linux
        return os.cpu_count()


def solve_one(instance, delta, eps, deadline, loop=False):
    """Run the algorithm on one instance, return a dict with the results (or None if the time budget is already reached)."""
    if deadline is not None and time.monotonic() >= deadline:
        return None  # budget reached before starting: skip this instance
    if not loop:
        t0 = time.perf_counter()
        best_sol, best_value, POS, num_t, status = approx_ordered(
            instance["cost_matrix"], instance["W"], delta=delta, eps=eps, deadline=deadline)
        return {
            "instance": instance["instance"], "best_sol": best_sol, "best_value": best_value,
            "p": len(POS), "num_t": num_t, "status": status, "time_s": time.perf_counter() - t0,
        }
    else:
        t0 = time.perf_counter()
        best_sol, best_value, POS, num_t, status = approx_ordered_loop(
            instance["cost_matrix"], instance["W"], delta=delta, eps=eps, deadline=deadline)
        return {
            "instance": instance["instance"], "best_sol": best_sol, "best_value": best_value,
            "p": len(POS), "num_t": num_t, "status": status, "time_s": time.perf_counter() - t0,
        }
 

 
def test_approx_assign(n, delta, eps, w_type, nb_instances, save=True, seed=0,
                       results_dir="../results/assignment", save_solutions=True,
                       n_workers=None, time_limit=None, rerun="unfinished", loop=False):
    path = results_path(results_dir, n, delta, eps, w_type, loop)
    done = set()
    if save:
        os.makedirs(results_dir, exist_ok=True)
        done = prepare_resume(path, rerun)

    instances = generate_instances(n, nb_instances, w_type, cost_seed=seed)
    todo = [inst for inst in instances if str(inst["instance"]) not in done]
    n_workers = n_workers or n_available_cores()
    deadline = time.monotonic() + time_limit if time_limit else None
    print(f"n={n}: {len(todo)} instance(s) to solve on {n_workers} worker(s), "
          f"budget {time_limit if time_limit else 'unlimited'} s", flush=True)

    rows, not_started = [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(solve_one, inst, delta, eps, deadline, loop) for inst in todo]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                not_started += 1
                continue

            row = {
                "method": "chen_appendixC",
                "n": n, "delta": delta, "eps": eps, "w_type": w_type, "seed": seed,
                "instance": res["instance"],
                "status": res["status"],
                "best_value": res["best_value"],
                "num_t": res["num_t"],
                "p": res["p"],
                "time_s": res["time_s"],
                "time_limit_total_s": time_limit,
                "n_workers": n_workers,
                "guarantee": (1 + delta) * (1 + 2 * eps),
            }
            if save_solutions:
                row["assignment"] = (" ".join(map(str, res["best_sol"]))
                                     if res["best_sol"] is not None else "")
            rows.append(row)

            if save:
                append_row(path, row)
            val = f"{res['best_value']:.2f}" if res["best_value"] is not None else "none"
            print(f"[chen_appendixC] n={n}, instance {res['instance']} {res['status']} "
                  f"(value={val}, {res['num_t']} vectors, {res['time_s']:.1f}s)", flush=True)

    if not_started:
        print(f"n={n}: time limit reached, {not_started} instance(s) not started", flush=True)

    return pd.read_csv(path) if save else pd.DataFrame(rows)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "--n-values", dest="n", type=int, nargs="+", default=[10])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--w-type", default="s_gini_1.5_weights")
    parser.add_argument("--nb-instances", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=55000.0,
                        help="wall-clock budget (s) for all instances of one n value; <= 0 means unlimited")
    parser.add_argument("--workers", type=int, default=None,
                        help="number of worker processes (default: cores allocated by OAR)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--rerun", choices=RERUN_MODES, default="unfinished")
    parser.add_argument("--loop", action="store_true")

    args = parser.parse_args()

    results_dir = os.path.dirname(args.output) if args.output else "../results/assignment"
    time_limit = args.time_limit if args.time_limit > 0 else None

    dfs = [test_approx_assign(n=n, delta=args.delta, eps=args.eps, w_type=args.w_type,
                              nb_instances=args.nb_instances, save=True, seed=args.seed,
                              results_dir=results_dir, n_workers=args.workers,
                              time_limit=time_limit, rerun=args.rerun)
           for n in args.n]

    if args.output:
        out = pd.concat(dfs, ignore_index=True)
        if args.output.endswith(".parquet"):
            out.to_parquet(args.output, index=False)
        else:
            out.to_csv(args.output, index=False)
        print(f"Results written to {args.output}")