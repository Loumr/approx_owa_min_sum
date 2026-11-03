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


# ----------------------------------------------------------------------
# Couts
# ----------------------------------------------------------------------
def generate_cost_matrices(n, nb_instances, seed=0):
    """Rejoue rng = np.random.default_rng(seed); puis
    rng.integers(1, 10**4, (n, n), dtype=int) dans l'ordre croissant
    d'instance -- meme logique que regenerate_instances.py."""
    rng = np.random.default_rng(seed)
    return [rng.integers(1, 10**4, (n, n), dtype=int).astype(float) for _ in range(nb_instances)]


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
