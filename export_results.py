"""
Export des résultats vers Excel : ordered_opt.py (approximation de Chen et al.)
et min_lorenz.py (exact Gurobi, min_lorenz, min_lorenz_v2).

- Résultats bruts : un fichier par configuration (méthode, poids, n, delta, eps),
  une ligne par instance.
- Résumés : un fichier avec une ligne par configuration, et optionnellement
  un fichier de résumé par configuration.
- Si des résultats exacts existent pour les mêmes (poids, n, graine, instance),
  chaque méthode reçoit la valeur exacte et le ratio valeur / exact. Quand l'exact
  n'a pas fini, les bornes de Gurobi donnent un encadrement garanti du ratio.
  Quand les deux lignes ont une empreinte d'instance, rien n'est calculé si elles diffèrent.

Fichiers lus (CSV écrits instance par instance) :
    approx_n{n}_delta{delta}_eps{eps}_{w_type}.csv       (ordered_opt.py)
    exact_n{n}_{w_type}.csv                              (min_lorenz.py --method exact)
    min_lorenz[_v2]_n{n}_eps{eps}_{w_type}.csv           (min_lorenz.py)

Usage :
    python export_results.py --results-dir ../results/assignment --out-dir ../results/excel
    python export_results.py --summary-per-config
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PATTERNS = ("approx_n*_delta*_eps*_*.csv", "exact_n*_*.csv", "min_lorenz*_n*_eps*_*.csv")
CONFIG_KEYS = ["method", "w_type", "n", "delta", "eps"]
INSTANCE_KEYS = CONFIG_KEYS + ["seed", "instance"]
MATCH_KEYS = ["w_type", "n", "seed", "instance"]   # pour retrouver l'exact d'une instance
EXACT_METHOD = "exact"

# Libellés lisibles (colonnes absentes ou entièrement vides ignorées)
RAW_LABELS = {
    "method": "Méthode",
    "w_type": "Poids",
    "n": "n",
    "delta": "delta",
    "eps": "eps",
    "seed": "Graine",
    "instance": "Instance",
    "instance_hash": "Empreinte instance",
    "status": "Statut",
    "best_value": "Valeur OWA",
    "exact_value": "Valeur exacte",
    "ratio_to_exact": "Ratio / exact",
    "exact_lower_bound": "Borne inf. optimum (Gurobi)",
    "exact_upper_bound": "Borne sup. optimum (Gurobi)",
    "ratio_bound_low": "Ratio ≥",
    "ratio_bound_high": "Ratio ≤ (garanti)",
    "same_instance": "Même instance que l'exact",
    "num_t": "Nb vecteurs t",
    "p": "p (|POS|)",
    "nb_sols": "Nb solutions distinctes",
    "mip_gap": "Gap MIP",
    "time_s": "Temps (s)",
    "time_limit_total_s": "Budget total (s)",
    "n_workers": "Nb workers",
    "guarantee": "Garantie (1+δ)(1+2ε)",
    "assignment": "Affectation",
}

FONT = "Arial"
HEADER_FILL = PatternFill("solid", start_color="D9E1F2")
TIMEOUT_FILL = PatternFill("solid", start_color="FCE4D6")


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def load_results(results_dir="../results/assignment", patterns=PATTERNS):
    """
    Charge et concatène tous les CSV de résultats d'un dossier.

    Les doublons (même configuration, graine et instance, par exemple après une
    reprise) sont supprimés en gardant la dernière ligne écrite. Dans les fichiers sans
    colonne 'status' (anciennes versions sans limite de temps), une ligne avec une valeur
    est considérée comme terminée.
    """
    files = sorted({f for p in patterns for f in glob.glob(os.path.join(results_dir, p))})
    if not files:
        raise FileNotFoundError(f"Aucun fichier de résultats ({', '.join(patterns)}) dans {results_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for col in CONFIG_KEYS:
        if col not in df.columns:
            df[col] = np.nan
    # Fichiers écrits par d'anciennes versions, sans colonne 'status' : ces versions n'avaient
    # pas de limite de temps et n'écrivaient une ligne qu'une fois l'instance terminée, donc
    # une ligne avec une valeur est une instance terminée.
    if "status" not in df.columns:
        df["status"] = np.nan
    no_status = df["status"].isna()
    df.loc[no_status & df["best_value"].notna(), "status"] = "completed"
    df["status"] = df["status"].fillna("unknown")
    if "assignment" in df.columns:
        df["assignment"] = df["assignment"].fillna("").astype(str)

    keys = [k for k in INSTANCE_KEYS if k in df.columns]
    df = df.drop_duplicates(subset=keys, keep="last")
    return sort_results(add_exact_reference(df))


def add_exact_reference(df):
    """
    Compare chaque ligne aux résultats exacts (Gurobi) de la même instance.

    Colonnes ajoutées :
        exact_value          optimum prouvé (exact au statut 'completed')
        ratio_to_exact       valeur / optimum
        exact_lower_bound    borne inférieure prouvée de l'optimum (ObjBound de Gurobi ;
                             l'optimum lui-même si l'exact est terminé)
        exact_upper_bound    meilleure solution trouvée par Gurobi (borne supérieure de l'optimum)
        ratio_bound_low      max(1, valeur / borne supérieure) : le vrai ratio est au moins égal à ceci
        ratio_bound_high     valeur / borne inférieure : le vrai ratio est au plus égal à ceci
        same_instance        concordance des empreintes d'instance

    Les bornes sont disponibles même quand l'exact a atteint la limite de temps. Si les
    empreintes des deux lignes diffèrent, rien n'est calculé.
    """
    df = df.copy()
    for col in ("exact_value", "ratio_to_exact", "exact_lower_bound", "exact_upper_bound",
                "ratio_bound_low", "ratio_bound_high"):
        df[col] = np.nan
    df["same_instance"] = ""

    exact = df[df["method"] == EXACT_METHOD].drop_duplicates(subset=MATCH_KEYS, keep="last")
    if exact.empty:
        return df

    completed = exact["status"] == "completed"
    lb = exact["lower_bound"] if "lower_bound" in exact else pd.Series(np.nan, index=exact.index)
    lb = lb.where(lb.notna(), exact["best_value"].where(completed))  # anciennes lignes sans borne
    ref = pd.DataFrame({
        **{k: exact[k] for k in MATCH_KEYS},
        "_exact_value": exact["best_value"].where(completed),
        "_exact_lb": lb,
        "_exact_ub": exact["best_value"],
        "_exact_hash": exact["instance_hash"] if "instance_hash" in exact else np.nan,
    })
    merged = df.merge(ref, on=MATCH_KEYS, how="left")

    has_ref = merged["_exact_lb"].notna() | merged["_exact_ub"].notna()
    both = merged["instance_hash"].notna() & merged["_exact_hash"].notna() \
        if "instance_hash" in merged else pd.Series(False, index=merged.index)
    same = ~both | (merged["instance_hash"] == merged["_exact_hash"])
    merged.loc[has_ref & both, "same_instance"] = np.where(same[has_ref & both], "oui", "NON")
    merged.loc[has_ref & ~both, "same_instance"] = "non vérifiable"

    ok = has_ref & same
    v = merged["best_value"]
    merged.loc[ok, "exact_value"] = merged.loc[ok, "_exact_value"]
    merged.loc[ok, "ratio_to_exact"] = v[ok] / merged.loc[ok, "_exact_value"]
    merged.loc[ok, "exact_lower_bound"] = merged.loc[ok, "_exact_lb"]
    merged.loc[ok, "exact_upper_bound"] = merged.loc[ok, "_exact_ub"]
    pos_lb = ok & (merged["_exact_lb"] > 0)
    merged.loc[pos_lb, "ratio_bound_high"] = v[pos_lb] / merged.loc[pos_lb, "_exact_lb"]
    pos_ub = ok & (merged["_exact_ub"] > 0)
    # Le vrai ratio vaut au moins 1 : on ne garde la borne que si elle apporte une information
    merged.loc[pos_ub, "ratio_bound_low"] = np.maximum(v[pos_ub] / merged.loc[pos_ub, "_exact_ub"], 1.0)
    return merged.drop(columns=["_exact_value", "_exact_lb", "_exact_ub", "_exact_hash"])


def _natural_key(value):
    """Tri 'naturel' : instance 2 avant instance 10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(value))]


def sort_results(df):
    keys = [k for k in CONFIG_KEYS + ["seed"] if k in df.columns]
    df = df.copy()
    df["_inst_order"] = df["instance"].map(_natural_key).map(
        lambda parts: tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in parts))
    df = df.sort_values(keys + ["_inst_order"], na_position="first").drop(columns="_inst_order")
    return df.reset_index(drop=True)


def _groups(df):
    """Regroupement par configuration, en gardant les clés vides (delta/eps sans objet)."""
    return df.groupby(CONFIG_KEYS, sort=True, dropna=False)


# ---------------------------------------------------------------------------
# Résumés
# ---------------------------------------------------------------------------

def _first(g, col):
    return g[col].iloc[0] if col in g and g[col].notna().any() else np.nan


def summarize(df, nb_instances=None):
    """Une ligne par configuration avec les statistiques sur les instances."""
    rows = []
    for key, g in _groups(df):
        t, v = g["time_s"], g["best_value"]
        row = dict(zip(CONFIG_KEYS, key))
        row.update({
            "Nb instances prévues": max(nb_instances, len(g)) if nb_instances else np.nan,
            "Nb instances": len(g),
            "Nb non démarrées": max(nb_instances - len(g), 0) if nb_instances else np.nan,
            "Nb terminées": int((g["status"] == "completed").sum()),
            "Nb timeout": int((g["status"] == "timeout").sum()),
            "Autre statut": int((~g["status"].isin(["completed", "timeout"])).sum()),
            "Valeur OWA moy.": v.mean(),
            "Valeur OWA méd.": v.median(),
            "Valeur OWA min": v.min(),
            "Valeur OWA max": v.max(),
            "Temps moy. (s)": t.mean(),
            "Temps méd. (s)": t.median(),
            "Temps écart-type (s)": t.std(ddof=1) if len(g) > 1 else np.nan,
            "Temps min (s)": t.min(),
            "Temps max (s)": t.max(),
            "Temps cumulé (s)": t.sum(),
        })
        r = g["ratio_to_exact"]
        row.update({
            "Nb instances avec exact": int(r.notna().sum()),
            "Ratio / exact moy.": r.mean(),
            "Ratio / exact max": r.max(),
            "Nb ratio = 1": int(np.isclose(r.dropna(), 1.0).sum()),
            "Empreinte ≠ exact": int((g["same_instance"] == "NON").sum()),
        })
        # Bornes garanties sur le ratio, y compris quand l'exact n'a pas fini
        rh, rl = g["ratio_bound_high"], g["ratio_bound_low"]
        row.update({
            "Nb instances avec borne": int(rh.notna().sum()),
            "Ratio ≤ (garanti) moy.": rh.mean(),
            "Ratio ≤ (garanti) max": rh.max(),
            "Ratio ≥ moy.": rl.mean(),
        })
        for col, label in (("num_t", "Nb vecteurs t"), ("nb_sols", "Nb solutions distinctes"),
                           ("mip_gap", "Gap MIP")):
            if col in g and g[col].notna().any():
                row[f"{label} moy."] = g[col].mean()
                row[f"{label} max"] = g[col].max()
        row["p (|POS|)"] = _first(g, "p")
        row["Garantie (1+δ)(1+2ε)"] = _first(g, "guarantee")
        row["Budget total (s)"] = _first(g, "time_limit_total_s")
        row["Nb workers"] = _first(g, "n_workers")
        rows.append(row)

    summary = pd.DataFrame(rows).dropna(axis=1, how="all")
    return summary.rename(columns={k: RAW_LABELS[k] for k in CONFIG_KEYS})


# ---------------------------------------------------------------------------
# Écriture Excel
# ---------------------------------------------------------------------------

def _config_name(method, w_type, n, delta, eps):
    parts = [str(method), f"n{int(n)}"]
    if pd.notna(delta):
        parts.append(f"delta{delta}")
    if pd.notna(eps):
        parts.append(f"eps{eps}")
    parts.append(str(w_type))
    return "_".join(parts)


def _format_sheet(ws, highlight_col=None, highlight_value="timeout"):
    """Police Arial, en-tête en gras, volets figés, filtres, largeurs et formats numériques."""
    headers = [c.value for c in ws[1]]
    for cell in ws[1]:
        cell.font = Font(name=FONT, bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 32

    hl_idx = headers.index(highlight_col) + 1 if highlight_col in headers else None

    for row in ws.iter_rows(min_row=2):
        fill = TIMEOUT_FILL if hl_idx and row[hl_idx - 1].value == highlight_value else None
        for cell, header in zip(row, headers):
            cell.font = Font(name=FONT)
            if fill:
                cell.fill = fill
            if isinstance(cell.value, float):
                h = str(header)
                if "Temps" in h or "Budget" in h:
                    cell.number_format = "0.00"
                elif "Garantie" in h or "Ratio" in h or "Gap" in h:
                    cell.number_format = "0.0000"
                else:
                    cell.number_format = "#,##0.00"

    for j, header in enumerate(headers, start=1):
        values = [str(header)] + [str(c.value) for c in ws[get_column_letter(j)][1:] if c.value is not None]
        width = min(max(len(s) for s in values) + 2, 45)
        ws.column_dimensions[get_column_letter(j)].width = max(width, 10)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def _write_excel(sheets, path, highlight_col=None):
    """sheets : dict {nom_onglet: DataFrame}."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    wb = load_workbook(path)
    for ws in wb.worksheets:
        _format_sheet(ws, highlight_col=highlight_col)
    wb.save(path)
    return path


def export_raw(df, out_dir="../results/excel/raw", include_assignment=True):
    """Un fichier Excel par configuration, une ligne par instance."""
    paths = []
    for key, g in _groups(df):
        cols = [c for c in RAW_LABELS if c in g.columns
                and (g[c].notna() & (g[c].astype(str) != "")).any()]
        if not include_assignment and "assignment" in cols:
            cols.remove("assignment")
        frame = g[cols].rename(columns=RAW_LABELS)
        path = os.path.join(out_dir, f"raw_{_config_name(*key)}.xlsx")
        paths.append(_write_excel({"Instances": frame}, path, highlight_col=RAW_LABELS["status"]))
    return paths


def export_summary(df, out_path="../results/excel/summary_all.xlsx", per_config_dir=None,
                   nb_instances=30):
    """
    Résumé global (une ligne par configuration), avec un onglet par type de poids
    s'il y en a plusieurs. Avec per_config_dir, écrit aussi un résumé par configuration.
    """
    summary = summarize(df, nb_instances)
    sheets = {"Toutes configurations": summary}
    if summary[RAW_LABELS["w_type"]].nunique() > 1:
        for w_type, g in summary.groupby(RAW_LABELS["w_type"]):
            sheets[str(w_type)] = g
    paths = [_write_excel(sheets, out_path)]

    if per_config_dir:
        labels = [RAW_LABELS[k] for k in CONFIG_KEYS]
        for _, row in summary.iterrows():
            name = _config_name(*(row.get(l, np.nan) for l in labels))
            vertical = row.dropna().reset_index()
            vertical.columns = ["Indicateur", "Valeur"]
            path = os.path.join(per_config_dir, f"summary_{name}.xlsx")
            paths.append(_write_excel({"Résumé": vertical}, path))
    return paths


def export_all(results_dir="../results/assignment", out_dir="../results/excel", nb_instances=30,
               summary_per_config=False, include_assignment=True):
    df = load_results(results_dir)
    raw = export_raw(df, os.path.join(out_dir, "raw"), include_assignment=include_assignment)
    summ = export_summary(df, os.path.join(out_dir, "summary_all.xlsx"), nb_instances=nb_instances,
                          per_config_dir=os.path.join(out_dir, "summary") if summary_per_config else None)
    print(f"{len(df)} ligne(s) lue(s), {_groups(df).ngroups} configuration(s)")
    print(f"{len(raw)} fichier(s) brut(s) dans {os.path.join(out_dir, 'raw')}")
    print(f"Résumé(s) : {', '.join(summ)}")
    mismatch = int((df["same_instance"] == "NON").sum())
    if mismatch:
        print(f"ATTENTION : {mismatch} ligne(s) dont l'empreinte diffère de celle de l'exact "
              f"(ratio non calculé) — voir la colonne « Même instance que l'exact »")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export des résultats vers Excel")
    parser.add_argument("--results-dir", default="../results/assignment")
    parser.add_argument("--out-dir", default="../results/excel")
    parser.add_argument("--summary-per-config", action="store_true",
                        help="écrit aussi un fichier de résumé par configuration")
    parser.add_argument("--nb-instances", type=int, default=30,
                        help="instances prévues par configuration ; 0 : celles trouvées")
    parser.add_argument("--no-assignment", action="store_true",
                        help="n'inclut pas la colonne d'affectation dans les fichiers bruts")
    args = parser.parse_args()

    export_all(args.results_dir, args.out_dir, summary_per_config=args.summary_per_config,
               include_assignment=not args.no_assignment, nb_instances=args.nb_instances or None)
