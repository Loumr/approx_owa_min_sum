"""
instance_generator.py

Regenere les instances exactement comme dans le pipeline original :
    - couts : rng = np.random.default_rng(seed) rejoue dans l'ordre
      croissant d'instance (meme logique que regenerate_instances.py) ;
    - poids : meme formules que le script de generation (linear, s_gini_*,
      sqrt, random) ;
    - k n'est PAS un parametre libre : il est DERIVE du regroupement
      geometrique approximate_weights(W, eps), exactement comme dans
      kfowama_benchmark.py. mu et lambda proviennent aussi de ce
      regroupement.
"""

import numpy as np


# Default parameters of the cost generators
FACTORS = [1, 2, 4]
PROP = [0.8, 0.15, 0.05]
COEF = 1
GEN_INST = ("uniform", "double_uniform", "heterogene", "same")


# ----------------------------------------------------------------------
# Couts
# ----------------------------------------------------------------------
def generate_cost_matrices(n, nb_instances, seed=0):
    """Rejoue rng = np.random.default_rng(seed); puis
    rng.integers(1, 10**4, (n, n), dtype=int) dans l'ordre croissant
    d'instance -- meme logique que regenerate_instances.py."""
    rng = np.random.default_rng(seed)
    return [rng.integers(1, 10**4, (n, n), dtype=int).astype(float) for _ in range(nb_instances)]


def generate_cost_matrices_double_uniform(n, nb_instances, prob, seed=0):
    rng = np.random.default_rng(seed)
    instances = []

    for _ in range(nb_instances):
        A = rng.integers(1, 1001, (n, n), dtype=int).astype(float)
        B = rng.integers(5000, 10000, (n, n), dtype=int).astype(float)

        mask = rng.choice([1, 0], (n, n), p=[prob, 1-prob])

        inst = A * mask + B * (1 - mask)

        instances.append(inst)

    return instances


def generate_cost_matrices_heterogene(n, nb_instances, factors, prop=[0.8, 0.15, 0.05], seed=0):
    if len(factors) != len(prop):
        raise ValueError("List of multipliers not the same size as the list of probabilities")
    
    rng = np.random.default_rng(seed)
    factors = np.asarray(factors, dtype=float)
    instances = []

    for _ in range(nb_instances):
        mask = rng.choice(len(prop), size=(n, n), p=prop)
        A = rng.integers(1, 1001, (n, n), dtype=int).astype(float)

        instances.append(A * factors[mask])

    return instances


def generate_cost_matrices_same(n, nb_instances, seed=0):
    rng = np.random.default_rng(seed)
    instances = []

    for _ in range(nb_instances):
        vect_costs = rng.integers(1, 1001, n, dtype=int).astype(float)
        A = np.tile(vect_costs, (n, n))
        instances.append(A)

    return instances




# ----------------------------------------------------------------------
# Poids
# ----------------------------------------------------------------------
def s_gini_weights(n, delta):
    k = np.arange(n, -1, -1)
    f = (k / n) ** delta
    return f[:-1] - f[1:]


def get_weights(w_type, n, instance):
    """w_type sous la forme 'linear_weights', 's_gini_1.5_weights',
    'random_weights', 'sqrt_weights' (meme convention que
    regenerate_instances.py)."""
    base = w_type.replace("_weights", "")

    if base == "linear":
        return np.arange(1, n + 1)[::-1] / (n * (n + 1))

    if base.startswith("s_gini"):
        delta = float(base.split("_")[2])
        return s_gini_weights(n, delta)

    if base == "sqrt":
        W = np.ones(n)
        W[0] = np.sqrt(n)
        return W

    if base == "random":
        BASE_SEED = 12345
        rng_i = np.random.default_rng(BASE_SEED + instance)
        rand = rng_i.uniform(0, 1, n)
        return np.sort(rand)[::-1]

    raise ValueError(f"w_type inconnu : {w_type}")


# ----------------------------------------------------------------------
# Instance complete : couts + (mu, lambda, k) derives de eps
# ----------------------------------------------------------------------
def generate_instances(n, nb_instances, w_type, cost_seed=0):
    """
    Genere nb_instances instances pour (n, w_type).

    Retourne une liste de dicts :
        {"instance": i, "cost_matrix": ..., "W": ...}
    """
    cost_matrices = generate_cost_matrices(n, nb_instances, seed=cost_seed)
    instances = []
    for i in range(nb_instances):
        W = get_weights(w_type, n, i)
        instances.append({
            "instance": i,
            "cost_matrix": cost_matrices[i],
            "W": W,
        })
    return instances


def generate_instances_double_uniform(n, nb_instances, w_type, prob, cost_seed=0):
    cost_matrices = generate_cost_matrices_double_uniform(n, nb_instances, prob, seed=cost_seed)
    instances = []
    for i in range(nb_instances):
        W = get_weights(w_type, n, i)
        instances.append({
            "instance": i,
            "cost_matrix": cost_matrices[i],
            "W": W,
        })
    return instances


def generate_instances_heterogene(n, nb_instances, w_type, factors, prop, cost_seed=0):
    cost_matrices = generate_cost_matrices_heterogene(n, nb_instances, factors, prop, seed=cost_seed)
    instances = []
    for i in range(nb_instances):
        W = get_weights(w_type, n, i)
        instances.append({
            "instance": i,
            "cost_matrix": cost_matrices[i],
            "W": W,
        })
    return instances



def generate_instances_same(n, nb_instances, w_type, cost_seed=0):
    cost_matrices = generate_cost_matrices_same(n, nb_instances, seed=cost_seed)
    instances = []
    for i in range(nb_instances):
        W = get_weights(w_type, n, i)
        instances.append({
            "instance": i,
            "cost_matrix": cost_matrices[i],
            "W": W,
        })
    return instances



def _fmt_list(values):
    """[1, 2, 4] -> '1-2-4' ; [0.8, 0.15, 0.05] -> '0.8-0.15-0.05'."""
    return "-".join(f"{v:g}" for v in values)


def instance_tag(gen_inst, factors=FACTORS, prop=PROP, coef=COEF):
    """Sub-folder name identifying the instance family AND its parameters."""
    if gen_inst == "uniform":
        return "uniform"
    if gen_inst == "double_uniform":
        return f"double_uniform_coef{coef:g}"
    if gen_inst == "heterogene":
        return f"heterogene_f{_fmt_list(factors)}_p{_fmt_list(prop)}"
    raise ValueError(f"unknown gen_inst: {gen_inst}")


def generate_instances_by_type(gen_inst, n, nb_instances, w_type, seed=0,
                               factors=FACTORS, prop=PROP, coef=COEF):
    """Same instances whatever the script that calls it."""
    if gen_inst == "uniform":
        return generate_instances(n, nb_instances, w_type, cost_seed=seed)
    if gen_inst == "double_uniform":
        prob = min(coef / n, 1.0)
        return generate_instances_double_uniform(n, nb_instances, w_type, prob, cost_seed=seed)
    if gen_inst == "heterogene":
        return generate_instances_heterogene(n, nb_instances, w_type, factors, prop, cost_seed=seed)
    if gen_inst == "same":
        return generate_instances_same(n, nb_instances, w_type, cost_seed=seed)

    raise ValueError(f"unknown gen_inst: {gen_inst}")