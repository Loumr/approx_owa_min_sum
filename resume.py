"""
resume.py

Reprise des expériences à partir des CSV de résultats (une ligne par instance),
partagé par ordered_opt.py et min_lorenz.py.

Modes de reprise (--rerun) :
    unfinished  (défaut) relance les instances absentes ET celles qui n'ont pas terminé
                (timeout, erreur du solveur, statut inconnu)
    all         relance toutes les instances
    none        ne relance que les instances absentes du fichier

Les lignes remplacées ne sont pas perdues : elles sont déplacées dans
<fichier>_replaced.csv, avec la date du remplacement.
"""

import os
import tempfile
from datetime import datetime

import pandas as pd

RERUN_MODES = ("unfinished", "all", "none")


def _statuses(df):
    """Statut de chaque ligne. Anciens fichiers sans colonne 'status' : ces versions n'avaient
    pas de limite de temps, donc une ligne avec une valeur est une instance terminée."""
    if "status" in df.columns:
        return df["status"].fillna("unknown")
    return df["best_value"].notna().map({True: "completed", False: "unknown"})


def _write_csv_atomic(df, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def prepare_resume(path, rerun="unfinished"):
    """
    Prépare la reprise d'un fichier de résultats et renvoie l'ensemble des instances
    (identifiants en str) à NE PAS relancer.

    Les lignes à relancer sont retirées du fichier (et archivées dans <fichier>_replaced.csv),
    pour qu'il ne reste qu'une ligne par instance.
    """
    if rerun not in RERUN_MODES:
        raise ValueError(f"rerun doit être parmi {RERUN_MODES}, reçu : {rerun}")
    if not os.path.exists(path):
        return set()

    df = pd.read_csv(path)
    if df.empty:
        return set()
    status = _statuses(df)

    if rerun == "none":
        keep = pd.Series(True, index=df.index)
    elif rerun == "all":
        keep = pd.Series(False, index=df.index)
    else:
        keep = status == "completed"

    # Une seule ligne conservée par instance (la dernière parmi celles gardées)
    ids = df["instance"].astype(str)
    last_kept = ids[keep].drop_duplicates(keep="last").index
    keep = pd.Series(df.index.isin(last_kept), index=df.index)

    removed = df[~keep]
    if len(removed):
        archive = path[:-4] + "_replaced.csv" if path.endswith(".csv") else path + "_replaced.csv"
        removed = removed.assign(replaced_at=datetime.now().isoformat(timespec="seconds"))
        if os.path.exists(archive):
            removed = pd.concat([pd.read_csv(archive), removed], ignore_index=True)
        _write_csv_atomic(removed, archive)
        _write_csv_atomic(df[keep], path)
        counts = _statuses(df[~keep]).value_counts().to_dict()
        print(f"Reprise ({rerun}) : {len(df) - int(keep.sum())} ligne(s) retirée(s) de {path} "
              f"pour être relancée(s) {counts} ; archivées dans {archive}")

    done = set(df.loc[keep, "instance"].astype(str))
    print(f"Reprise ({rerun}) : {len(done)} instance(s) conservée(s) dans {path}")
    return done


def append_row(path, row):
    """
    Ajoute une ligne de résultats au CSV. Si les colonnes du fichier existant diffèrent
    (fichier écrit par une ancienne version du code), le fichier est réécrit avec l'union
    des colonnes au lieu d'ajouter une ligne décalée.
    """
    new = pd.DataFrame([row])
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        new.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    if header == list(new.columns):
        new.to_csv(path, mode="a", index=False, header=False)
    else:
        _write_csv_atomic(pd.concat([pd.read_csv(path), new], ignore_index=True), path)
