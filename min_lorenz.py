"""
min_lorenz.py

Heuristique "min Lorenz" (v1 et v2) et solveur exact (Gurobi) pour l'affectation OWA,
sur les mêmes instances que ordered_opt.py (instance_generator.generate_instances).

Méthodes (--method) :
    exact          MIP Gurobi (formulation duale de l'OWA)
    min_lorenz     seuils t, coûts max(c, t)
    min_lorenz_v2  seuils t, coûts max(c - t, 0), arrêt anticipé

--eps > 0 arrondit géométriquement les coûts ((1+eps)^floor(log_{1+eps} c)) avant de lancer
min_lorenz ; la valeur rapportée est toujours l'OWA de la solution sur les VRAIS coûts.

Sorties (dans --results-dir), une ligne par instance, écrite dès qu'elle est terminée :
    exact_n{n}_{w_type}.csv
    {method}_n{n}_eps{eps}_{w_type}.csv
Instances sauvegardées dans --instances-dir (voir instance_store.py).
"""

import os

# AVANT l'import de numpy : un seul thread de calcul par worker
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from instance_generator import generate_instances
from instance_store import instance_fingerprint, save_instances
from resume import RERUN_MODES, append_row, prepare_resume

METHODS = ("exact", "min_lorenz", "min_lorenz_v2")


class TimeLimitReached(Exception):
    pass


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeLimitReached


def owa_value(costs_vec, W):
    """OWA : coûts triés par ordre décroissant, pondérés par W (décroissant)."""
    return float(np.dot(W, np.sort(costs_vec)[::-1]))


def round_costs(cost_matrix, eps):
    """Arrondi géométrique : c -> (1+eps)^floor(log_{1+eps}(c)). Coûts supposés >= 1."""
    return (1 + eps) ** np.floor(np.log(cost_matrix) / np.log(1 + eps))


# ---------------------------------------------------------------------------
# Algorithmes
# ---------------------------------------------------------------------------

def min_lorenz(cost_matrix, W, variant="v1", deadline=None, state=None):
    """
    Parcourt les seuils t (valeurs distinctes des coûts) et résout une affectation
    min-somme sur max(c, t) (v1) ou max(c - t, 0) (v2). Garde la solution de plus
    petit OWA calculé sur cost_matrix.

    state (dict) reçoit la meilleure solution au fil de l'eau, pour qu'elle reste
    disponible si TimeLimitReached interrompt la boucle.
    Renvoie (affectation, nombre de solutions distinctes rencontrées).
    """
    state = {} if state is None else state
    costs = np.asarray(cost_matrix, dtype=float)
    rows = np.arange(costs.shape[0])

    _, assign = linear_sum_assignment(costs)
    best = owa_value(costs[rows, assign], W)
    unique = {tuple(assign)}
    state.update(best_sol=assign, unique=unique)

    for t in np.unique(costs)[1:]:
        _check_deadline(deadline)
        temp = np.maximum(costs, t) if variant == "v1" else np.maximum(costs - t, 0)
        _, assign_t = linear_sum_assignment(temp)
        unique.add(tuple(assign_t))

        value_t = owa_value(costs[rows, assign_t], W)
        if value_t < best:
            assign, best = assign_t, value_t
            state["best_sol"] = assign

        # v2 : toutes les affectations ont un coût nul au-delà de ce seuil
        if variant == "v2" and temp[rows, assign_t].sum() <= 1e-9:
            break

    return assign, len(unique)


def exact_gurobi(cost_matrix, W, time_limit=None, threads=1, mem_limit_gb=None, legacy_params=False):
    """
    OWA exact par MIP (formulation duale : alpha_i + beta_k >= W_i z_k, min sum alpha + beta).

    Renvoie un dict :
        assign       affectation (None si aucune solution trouvée)
        status       "completed" (optimum prouvé), "timeout", ou "gurobi_status_<code>"
        mip_gap      gap relatif final
        solver_obj   valeur de la meilleure solution selon Gurobi (borne supérieure de l'optimum)
        lower_bound  borne inférieure prouvée de l'optimum (ObjBound)
    """
    import gurobipy as gp
    from gurobipy import GRB

    C = np.asarray(cost_matrix, dtype=float)
    W = np.asarray(W, dtype=float)
    n = C.shape[0]

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("owa_assignment", env=env) as m:
        m.Params.Threads = threads        # 1 : les instances tournent déjà en parallèle
        m.Params.MIPGap = 0.0             # optimum prouvé, pas seulement à 0,01 % près
        if mem_limit_gb is not None:
            m.Params.MemLimit = mem_limit_gb   # en Go
        if legacy_params:                 # paramètres de la version d'origine
            m.Params.Method = 1
            m.Params.Presolve = 0
            m.Params.Crossover = 0
            m.Params.NumericFocus = 3
        if time_limit is not None:
            m.Params.TimeLimit = max(float(time_limit), 1e-3)

        x = m.addMVar((n, n), vtype=GRB.BINARY, name="x")
        z = m.addMVar(n, lb=0.0, name="z")
        alpha = m.addMVar(n, lb=-GRB.INFINITY, name="alpha")
        beta = m.addMVar(n, lb=-GRB.INFINITY, name="beta")

        for i in range(n):
            m.addConstr(z[i] == C[i] @ x[i, :])
            m.addConstr(x[i, :].sum() == 1)
            m.addConstr(x[:, i].sum() == 1)
        for i in range(n):
            m.addConstr(alpha[i] + beta - W[i] * z >= 0)

        m.setObjective(alpha.sum() + beta.sum(), GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            status = "completed"
        elif m.Status == GRB.TIME_LIMIT:
            status = "timeout"
        else:
            status = f"gurobi_status_{m.Status}"

        # La borne existe même sans solution, dès que la relaxation à la racine est résolue
        try:
            lower_bound = float(m.ObjBound)
            if not np.isfinite(lower_bound):
                lower_bound = np.nan
        except gp.GurobiError:
            lower_bound = np.nan

        out = {"assign": None, "status": status, "mip_gap": np.nan,
               "solver_obj": np.nan, "lower_bound": lower_bound}
        if m.SolCount > 0:
            # Arrondi (et non troncature) : 0.9999999 doit donner 1
            out["assign"] = np.argmax(np.rint(x.X), axis=1)
            out["solver_obj"] = float(m.ObjVal)
            out["mip_gap"] = float(m.MIPGap)
        return out


# ---------------------------------------------------------------------------
# Expérience
# ---------------------------------------------------------------------------

def results_path(results_dir, method, n, eps, w_type):
    if method == "exact":
        return os.path.join(results_dir, f"exact_n{n}_{w_type}.csv")
    return os.path.join(results_dir, f"{method}_n{n}_eps{eps}_{w_type}.csv")


def n_available_cores():
    """Cœurs réellement alloués (cpuset OAR), et non tous les cœurs du nœud."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def solve_one(instance, method, eps, deadline, gurobi_threads, gurobi_mem_gb=None, gurobi_legacy=False):
    if deadline is not None and time.monotonic() >= deadline:
        return None  # budget épuisé avant le démarrage

    C, W = instance["cost_matrix"], instance["W"]
    rows = np.arange(C.shape[0])
    out = {"instance": instance["instance"], "instance_hash": instance_fingerprint(C, W),
           "nb_sols": np.nan, "mip_gap": np.nan, "solver_obj": np.nan, "lower_bound": np.nan}
    t0 = time.perf_counter()

    if method == "exact":
        remaining = None if deadline is None else deadline - time.monotonic()
        res = exact_gurobi(C, W, time_limit=remaining, threads=gurobi_threads,
                           mem_limit_gb=gurobi_mem_gb, legacy_params=gurobi_legacy)
        assign, status = res["assign"], res["status"]
        out.update(mip_gap=res["mip_gap"], solver_obj=res["solver_obj"],
                   lower_bound=res["lower_bound"])
    else:
        variant = "v1" if method == "min_lorenz" else "v2"
        work = round_costs(C, eps) if eps > 0 else C
        state = {}
        try:
            assign, nb_sols = min_lorenz(work, W, variant=variant, deadline=deadline, state=state)
            status = "completed"
        except TimeLimitReached:
            assign, nb_sols, status = state.get("best_sol"), len(state.get("unique", ())), "timeout"
        out["nb_sols"] = nb_sols

    out["time_s"] = time.perf_counter() - t0
    out["status"] = status
    out["best_sol"] = assign
    out["best_value"] = owa_value(C[rows, assign], W) if assign is not None else np.nan
    return out


def run(method, n, eps, w_type, nb_instances, seed=0, time_limit=None, n_workers=None,
        results_dir="../results/assignment", instances_dir="../results/instances",
        gurobi_threads=1, gurobi_mem_gb=None, gurobi_legacy=False, save_solutions=True,
        rerun="unfinished"):
    if method not in METHODS:
        raise ValueError(f"method doit être parmi {METHODS}, reçu : {method}")
    if method == "exact":
        eps = np.nan  # sans objet pour l'exact

    path = results_path(results_dir, method, n, eps, w_type)
    os.makedirs(results_dir, exist_ok=True)
    done = prepare_resume(path, rerun)

    instances = generate_instances(n, nb_instances, w_type, cost_seed=seed)
    save_instances(instances, n, w_type, seed, instances_dir)

    todo = [inst for inst in instances if str(inst["instance"]) not in done]
    n_workers = n_workers or n_available_cores()
    deadline = time.monotonic() + time_limit if time_limit else None
    print(f"{method} n={n} eps={eps} {w_type} : {len(todo)} instance(s) sur {n_workers} worker(s), "
          f"budget {time_limit if time_limit else 'illimité'} s", flush=True)

    not_started = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(solve_one, inst, method, eps, deadline, gurobi_threads,
                             gurobi_mem_gb, gurobi_legacy) for inst in todo]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                not_started += 1
                continue
            row = {
                "method": method, "n": n, "delta": np.nan, "eps": eps, "w_type": w_type,
                "seed": seed, "instance": res["instance"], "instance_hash": res["instance_hash"],
                "status": res["status"], "best_value": res["best_value"],
                "nb_sols": res["nb_sols"], "mip_gap": res["mip_gap"], "solver_obj": res["solver_obj"],
                "lower_bound": res["lower_bound"],
                "time_s": res["time_s"], "time_limit_total_s": time_limit, "n_workers": n_workers,
            }
            if save_solutions:
                row["assignment"] = (" ".join(map(str, res["best_sol"]))
                                     if res["best_sol"] is not None else "")
            append_row(path, row)
            print(f"[{method}] n={n}, instance {res['instance']} {res['status']} "
                  f"(valeur={res['best_value']:.2f}, {res['time_s']:.1f}s)", flush=True)

    if not_started:
        print(f"Budget épuisé : {not_started} instance(s) non démarrée(s)", flush=True)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="min Lorenz / exact Gurobi pour l'affectation OWA")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--eps", type=float, default=0.0,
                        help="arrondi géométrique des coûts pour min_lorenz (0 = pas d'arrondi)")
    parser.add_argument("--w-type", default="linear_weights")
    parser.add_argument("--nb-instances", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=3600.0,
                        help="budget total (s, temps réel) pour les instances d'un n ; <= 0 : illimité")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    parser.add_argument("--gurobi-mem-gb", type=float, default=None,
                        help="limite mémoire Gurobi par worker, en Go (défaut : aucune)")
    parser.add_argument("--gurobi-legacy-params", action="store_true",
                        help="Method=1, Presolve=0, Crossover=0, NumericFocus=3 (version d'origine)")
    parser.add_argument("--rerun", choices=RERUN_MODES, default="unfinished",
                        help="unfinished : relance les instances absentes ou non terminées ; "
                             "all : tout relancer ; none : seulement les absentes")
    parser.add_argument("--results-dir", default="../results/assignment")
    parser.add_argument("--instances-dir", default="../results/instances")
    args = parser.parse_args()

    for n in args.n:
        run(args.method, n, args.eps, args.w_type, args.nb_instances, seed=args.seed,
            time_limit=args.time_limit if args.time_limit > 0 else None,
            n_workers=args.workers, results_dir=args.results_dir,
            instances_dir=args.instances_dir, gurobi_threads=args.gurobi_threads,
            gurobi_mem_gb=args.gurobi_mem_gb, gurobi_legacy=args.gurobi_legacy_params,
            rerun=args.rerun)
