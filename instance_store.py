"""
instance_store.py

Sauvegarde des instances générées et empreinte (hash) de chaque instance,
pour pouvoir vérifier que deux méthodes ont bien tourné sur les mêmes données.

Un fichier par (n, w_type, seed) :
    <instances_dir>/instances_n{n}_{w_type}_seed{seed}.npz
contenant :
    instance  (nb,)       identifiants d'instance
    costs     (nb, n, n)  matrices de coûts (entiers, stockés en int32)
    W         (nb, n)     vecteurs de poids
    hash      (nb,)       empreinte de chaque instance
"""

import hashlib
import os
import tempfile
import warnings

import numpy as np


def instance_fingerprint(cost_matrix, W):
    """Empreinte courte (12 caractères hexadécimaux) d'une instance (coûts + poids)."""
    h = hashlib.sha1(np.ascontiguousarray(cost_matrix, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(W, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def instances_path(instances_dir, n, w_type, seed):
    return os.path.join(instances_dir, f"instances_n{n}_{w_type}_seed{seed}.npz")


def _write_npz_atomic(path, **arrays):
    """Écriture atomique : plusieurs jobs peuvent viser le même fichier en même temps."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_instances(path):
    """Recharge un fichier d'instances sous forme de liste de dicts (comme generate_instances)."""
    data = np.load(path)
    return [
        {"instance": int(i), "cost_matrix": data["costs"][k].astype(float),
         "W": data["W"][k], "hash": str(data["hash"][k])}
        for k, i in enumerate(data["instance"])
    ]


def save_instances(instances, n, w_type, seed, instances_dir="../results/instances"):
    """
    Sauvegarde les instances et renvoie {instance: hash}.

    Si le fichier existe déjà, les empreintes des instances communes sont comparées :
    une différence signifie que la génération a changé, et le fichier existant n'est
    PAS écrasé (un avertissement est émis). Si les empreintes concordent et que la
    nouvelle liste contient plus d'instances, le fichier est complété.
    """
    hashes = {int(inst["instance"]): instance_fingerprint(inst["cost_matrix"], inst["W"])
              for inst in instances}
    path = instances_path(instances_dir, n, w_type, seed)

    if os.path.exists(path):
        old = np.load(path)
        old_hashes = dict(zip(old["instance"].astype(int), old["hash"].astype(str)))
        mismatched = sorted(i for i in hashes if i in old_hashes and old_hashes[i] != hashes[i])
        if mismatched:
            warnings.warn(
                f"{path} : les instances {mismatched} ne correspondent pas aux instances "
                f"déjà sauvegardées. La génération a changé ; fichier existant conservé.",
                RuntimeWarning, stacklevel=2)
            return hashes
        if set(hashes) <= set(old_hashes):
            return hashes  # rien de nouveau

    ordered = sorted(instances, key=lambda inst: int(inst["instance"]))
    costs = np.stack([np.asarray(inst["cost_matrix"], dtype=float) for inst in ordered])
    if np.all(costs == np.round(costs)) and np.abs(costs).max() < 2**31:
        costs = costs.astype(np.int32)  # coûts entiers : stockage compact, sans perte
    _write_npz_atomic(
        path,
        instance=np.array([int(inst["instance"]) for inst in ordered]),
        costs=costs,
        W=np.stack([np.asarray(inst["W"], dtype=float) for inst in ordered]),
        hash=np.array([hashes[int(inst["instance"])] for inst in ordered]),
    )
    return hashes
