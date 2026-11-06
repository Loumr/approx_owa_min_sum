"""
export_latex.py

LaTeX tables (in English) comparing the methods on running times and approximation ratios,
built from the same CSV files as export_results.py.

Ratio of a method on an instance = OWA value / optimum.
    - If the exact solver proved optimality on that instance, the true ratio is used.
    - Otherwise, if Gurobi hit its time limit, the guaranteed upper bound
      value / (Gurobi lower bound) is used and the cell is marked with a dagger.

Required LaTeX package: booktabs (and graphicx if resize=True).

Usage:
    from export_latex import latex_comparison_table
    tex = latex_comparison_table("../results/assignment", out_path="../results/tables.tex")

    python export_latex.py --results-dir ../results/assignment --out ../results/tables.tex
    python export_latex.py --layout long --metrics time ratio_mean solved
"""

import argparse
import os

import numpy as np
import pandas as pd

from export_results import load_results

METRICS = {
    "time": "Avg. time (s)",
    "time_median": "Med. time (s)",
    "time_max": "Max. time (s)",
    "ratio_mean": "Avg. ratio",
    "ratio_max": "Max. ratio",
    "ratio": "Avg. (max) ratio",      # one column "1.0004 (1.0017)" instead of two
    "solved": "Solved",
}
SHORT_HEADERS = {
    "time": "Time", "time_median": "Med.", "time_max": "Max", "ratio_mean": "Ratio",
    "ratio_max": "Max", "ratio": "Ratio", "solved": "Sol.",
}
LOWER_IS_BETTER = {"time", "time_median", "time_max", "ratio_mean", "ratio_max", "ratio"}
RATIO_METRICS = {"ratio_mean", "ratio_max", "ratio"}

DAGGER, DDAGGER = r"$^\dagger$", r"$^\ddagger$"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _fmt_param(x):
    return f"{x:g}"


def method_label(method, delta=np.nan, eps=np.nan):
    """Readable (LaTeX) name of a method and its parameters."""
    if method == "exact":
        return "Exact (Gurobi)"
    if method == "chen_appendixC":
        return rf"Chen et al. ($\delta={_fmt_param(delta)}$, $\varepsilon={_fmt_param(eps)}$)"
    if method in ("min_lorenz", "min_lorenz_v2"):
        name = "Min-Lorenz" + (" v2" if method == "min_lorenz_v2" else "")
        if pd.notna(eps) and eps > 0:
            name += rf" ($\varepsilon={_fmt_param(eps)}$)"
        return name
    return _escape(str(method))


def weights_label(w_type):
    base = str(w_type).replace("_weights", "")
    if base.startswith("s_gini"):
        return f"S-Gini weights (parameter {base.split('_')[2]})"
    return {"linear": "linear weights", "sqrt": "square-root weights",
            "random": "random weights"}.get(base, _escape(base) + " weights")


def _escape(text):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def _method_order(method):
    return {"exact": 0, "chen_appendixC": 1, "min_lorenz": 2, "min_lorenz_v2": 3}.get(method, 9)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(g, nb_instances=None):
    """Statistics of one method on one (weights, n) cell."""
    exact_ratio = g["ratio_to_exact"]
    bound = g["ratio_bound_high"]
    ratio = exact_ratio.where(exact_ratio.notna(), bound)
    return {
        "time": g["time_s"].mean(),
        "time_median": g["time_s"].median(),
        "time_max": g["time_s"].max(),
        "ratio_mean": ratio.mean() if ratio.notna().any() else np.nan,
        "ratio_max": ratio.max() if ratio.notna().any() else np.nan,
        "n_solved": int((g["status"] == "completed").sum()),
        "n_started": len(g),
        # Denominator: number of instances of the experiment (not only those started)
        "n_total": max(nb_instances, len(g)) if nb_instances else len(g),
        "ratio_from_bound": bool((exact_ratio.isna() & bound.notna()).any()),
        "ratio_partial": 0 < ratio.notna().sum() < len(g),
        "is_exact": g["method"].iloc[0] == "exact",
    }


def compute_comparison(df, methods=None, nb_instances=None):
    """
    One row per (w_type, n, method variant) with the statistics used in the tables.
    methods: optional list of method names to keep (e.g. ["exact", "min_lorenz"]).
    nb_instances: number of instances of the experiment (denominator of "Solved").
    """
    if methods is not None:
        df = df[df["method"].isin(methods)]
    rows = []
    for (w_type, n, method, delta, eps), g in df.groupby(
            ["w_type", "n", "method", "delta", "eps"], dropna=False, sort=True):
        row = {"w_type": w_type, "n": int(n), "method": method, "delta": delta, "eps": eps,
               "label": method_label(method, delta, eps)}
        row.update(_stats(g, nb_instances))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out["method"].map(_method_order)
    return out.sort_values(["w_type", "n", "_order", "delta", "eps"]).drop(columns="_order")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_time(x, decimals):
    if pd.isna(x):
        return "--"
    if x >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    if x > 0 and x < 10 ** (-decimals):
        return f"$<{10 ** (-decimals):g}$"
    return f"{x:.{decimals}f}"


def _cell(stats, metric, time_decimals, ratio_decimals):
    """Returns (text, value used for 'best' comparison or None)."""
    if metric == "solved":
        return f"{stats['n_solved']}/{stats['n_total']}", None
    if metric in RATIO_METRICS:
        value = stats["ratio_mean" if metric == "ratio" else metric]
        if stats["is_exact"] or pd.isna(value):
            return "--", None
        text = f"{value:.{ratio_decimals}f}"
        if metric == "ratio":
            text += f" ({stats['ratio_max']:.{ratio_decimals}f})"
        if stats["ratio_from_bound"]:
            text += DAGGER
        if stats["ratio_partial"]:
            text += DDAGGER
        return text, value
    value = stats[metric]
    return _fmt_time(value, time_decimals), value


def _best_values(cells_by_variant, metric):
    values = [v for _, v in cells_by_variant if v is not None and pd.notna(v)]
    return min(values) if values and metric in LOWER_IS_BETTER else None


def _is_best(value, best, metric, time_decimals, ratio_decimals):
    """Compared on the values as displayed, so that ties after rounding are all in bold."""
    if best is None or value is None or pd.isna(value):
        return False
    d = ratio_decimals if metric in RATIO_METRICS else time_decimals
    return round(value, d) <= round(best, d)


def _bold(text):
    # The dagger marks stay outside \textbf
    for mark in (DAGGER + DDAGGER, DAGGER, DDAGGER):
        if text.endswith(mark):
            return rf"\textbf{{\boldmath {text[:-len(mark)]}}}{mark}"
    return rf"\textbf{{\boldmath {text}}}"   # \boldmath: also bolds the math parts (e.g. $<0.01$)


def _notes(comp, metrics, short_headers=False):
    notes = []
    if short_headers and any(m.startswith("time") for m in metrics):
        notes.append("Times in seconds" + (" (Med.: median)." if "time_median" in metrics else "."))
    if any(m in RATIO_METRICS for m in metrics):
        notes.append("Ratio: OWA value of the method divided by the optimal value (exact solver)"
                     + ("; average (maximum) over the instances." if "ratio" in metrics else "."))
        if comp["ratio_from_bound"].any():
            notes.append(DAGGER + " Upper bound on the ratio, computed with the lower bound "
                         "returned by Gurobi when it hit its time limit.")
        if comp["ratio_partial"].any():
            notes.append(DDAGGER + " Computed on the instances for which a reference value is available.")
    if any(m.startswith("time") for m in metrics) and (comp["n_solved"] < comp["n_started"]).any():
        notes.append("Times include the instances stopped at the time limit.")
    if (comp["n_started"] < comp["n_total"]).any():
        notes.append("Averages are computed over the instances that were started; instances "
                     "not started within the time budget are counted as unsolved.")
    if "solved" in metrics:
        notes.append(("Sol." if short_headers else "Solved") + ": instances completed within the time limit "
                     "(optimality proved for the exact solver).")
    return notes


def _header(metric, short):
    return SHORT_HEADERS[metric] if short else METRICS[metric]


def _two_lines(label):
    """'Chen et al. ($\\delta=0.1$, ...)' -> name and parameters on two lines (no extra package)."""
    if " (" not in label:
        return label
    name, params = label.split(" (", 1)
    return rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{name}\\({params}\end{{tabular}}"


def _format_row_cells(cells, best, metrics, j, time_decimals, ratio_decimals, bold_best):
    out = []
    for m in metrics:
        text, value = cells[m][j]
        if bold_best and _is_best(value, best.get(m), m, time_decimals, ratio_decimals):
            if m == "ratio" and " (" in text:   # only the average in bold: narrower
                head, tail = text.split(" (", 1)
                text = _bold(head) + " (" + tail
            else:
                text = _bold(text)
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _table_wide(comp, metrics, time_decimals, ratio_decimals, bold_best,
                short_headers=False, two_line_names=True):
    """
    Rows: n. Column groups: methods, each with one column per metric.
    The exact method gets no ratio columns (its ratio is 1 by definition).
    """
    first = comp.drop_duplicates("label")
    variants = first["label"].tolist()
    exact_labels = set(first.loc[first["is_exact"], "label"])
    vmetrics = {v: [m for m in metrics if not (v in exact_labels and m in RATIO_METRICS)]
                for v in variants}
    variants = [v for v in variants if vmetrics[v]]

    spec = "r" + "".join("r" * len(vmetrics[v]) for v in variants)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    header, rules, sub, col = [""], [], ["$n$"], 2
    for v in variants:
        k = len(vmetrics[v])
        name = _two_lines(v) if two_line_names else v
        header.append(rf"\multicolumn{{{k}}}{{c}}{{{name}}}")
        rules.append(rf"\cmidrule(lr){{{col}-{col + k - 1}}}")
        sub += [_header(m, short_headers) for m in vmetrics[v]]
        col += k
    lines += [" & ".join(header) + r" \\", " ".join(rules), " & ".join(sub) + r" \\", r"\midrule"]

    for n, g in comp.groupby("n", sort=True):
        by_label = {r["label"]: r for _, r in g.iterrows()}
        cells = {m: [(_cell(by_label[v], m, time_decimals, ratio_decimals) if v in by_label
                      else ("--", None)) for v in variants] for m in metrics}
        best = {m: _best_values(cells[m], m) for m in metrics} if bold_best else {}
        row = [str(n)]
        for j, v in enumerate(variants):
            row += _format_row_cells(cells, best, vmetrics[v], j, time_decimals, ratio_decimals, bold_best)
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines, _notes(comp, metrics, short_headers)


def _table_long(comp, metrics, time_decimals, ratio_decimals, bold_best,
                short_headers=False, two_line_names=False):
    """Rows: (n, method). Columns: metrics. Better when there are many methods."""
    lines = [rf"\begin{{tabular}}{{rl{'r' * len(metrics)}}}", r"\toprule",
             " & ".join(["$n$", "Method"] + [_header(m, short_headers) for m in metrics]) + r" \\", r"\midrule"]

    groups = list(comp.groupby("n", sort=True))
    for idx, (n, g) in enumerate(groups):
        cells = {m: [_cell(r, m, time_decimals, ratio_decimals) for _, r in g.iterrows()] for m in metrics}
        best = {m: _best_values(cells[m], m) for m in metrics} if bold_best else {}
        for j, (_, r) in enumerate(g.iterrows()):
            row = [str(n) if j == 0 else "", r["label"]]
            row += _format_row_cells(cells, best, metrics, j, time_decimals, ratio_decimals, bold_best)
            lines.append(" & ".join(row) + r" \\")
        if idx < len(groups) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines, _notes(comp, metrics, short_headers)


def latex_comparison_table(results, out_path=None, layout="wide",
                           metrics=("time", "ratio_mean", "ratio_max", "solved"),
                           methods=None, w_types=None, n_values=None, nb_instances=30,
                           time_decimals=2, ratio_decimals=4, bold_best=True, resize=False,
                           font_size="small", tabcolsep=None, short_headers=False,
                           two_line_names=True, landscape=False,
                           caption="Comparison of the methods on {weights}.",
                           label="tab:comparison_{w_type}"):
    """
    Build one LaTeX table per weight type comparing the methods.

    results     results directory (str) or DataFrame returned by export_results.load_results
    out_path    if given, the LaTeX code is also written to this file
    layout      "wide": one row per n, one column group per method
                "long": one row per (n, method), one column per metric
    metrics     among: time, time_median, time_max, ratio_mean, ratio_max, solved
    nb_instances    number of instances per configuration, used as denominator of "Solved"
                (None: number of instances found in the results)
    methods     method names to keep, e.g. ["exact", "chen_appendixC", "min_lorenz"] (default: all)
    w_types     weight types to keep (default: all); n_values: values of n to keep (default: all)
    bold_best   bold the best (smallest) time and ratio of each n
    resize      wrap the tabular in \\resizebox{\\textwidth}{!}{...} (needs graphicx)
    font_size   LaTeX size command without backslash: "small", "footnotesize", "scriptsize"...
    tabcolsep   space between columns, e.g. "3pt" (LaTeX default: 6pt)
    short_headers   short column headers (Time, Ratio, Max, Sol.), units given in a note
    two_line_names  (wide layout) method name and parameters on two lines in the header
    landscape   use sidewaystable (needs \\usepackage{rotating}): table rotated on its page
    caption, label  may use {weights} (readable weight name) and {w_type}

    Returns the LaTeX code (str).
    """
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; available: {list(METRICS)}")
    if layout not in ("wide", "long"):
        raise ValueError("layout must be 'wide' or 'long'")

    df = load_results(results) if isinstance(results, str) else results
    if w_types is not None:
        df = df[df["w_type"].isin(w_types)]
    if n_values is not None:
        df = df[df["n"].isin(n_values)]
    comp = compute_comparison(df, methods, nb_instances)
    if comp.empty:
        raise ValueError("No results match the selection")

    build = _table_wide if layout == "wide" else _table_long
    blocks = []
    for w_type, c in comp.groupby("w_type", sort=True):
        safe_w = str(w_type).replace(".", "-")
        tabular, notes = build(c, list(metrics), time_decimals, ratio_decimals, bold_best,
                               short_headers=short_headers, two_line_names=two_line_names)
        if resize:
            tabular = [r"\resizebox{\textwidth}{!}{%"] + tabular + ["}"]
        if notes:  # below the table, over the full text width
            tabular += [r"\par\smallskip", r"{\footnotesize\raggedright"] \
                + [note + r"\\" for note in notes[:-1]] + [notes[-1] + r"\par}"]
        caption_text = caption.format(weights=weights_label(w_type), w_type=_escape(str(w_type)))
        blocks.append("\n".join(
            [r"\begin{sidewaystable}" if landscape else r"\begin{table}[htbp]", r"\centering",
             "\\" + font_size]
            + ([rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else [])
            + [
             rf"\caption{{{caption_text}}}",   # above the table, as usual for tables
             rf"\label{{{label.format(weights=w_type, w_type=safe_w)}}}"]
            + tabular
            + [r"\end{sidewaystable}" if landscape else r"\end{table}"]))

    packages = ["booktabs"] + (["graphicx"] if resize else []) + (["rotating"] if landscape else [])
    tex = "% Requires " + ", ".join(rf"\usepackage{{{p}}}" for p in packages) + "\n\n" + "\n\n".join(blocks) + "\n"
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
    return tex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaTeX tables comparing the methods")
    parser.add_argument("--results-dir", default="../results/assignment")
    parser.add_argument("--out", default="../results/latex/comparison.tex")
    parser.add_argument("--layout", choices=["wide", "long"], default="wide")
    parser.add_argument("--metrics", nargs="+", default=["time", "ratio_mean", "ratio_max", "solved"],
                        choices=list(METRICS))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--w-types", nargs="+", default=None)
    parser.add_argument("--n", type=int, nargs="+", default=None)
    parser.add_argument("--nb-instances", type=int, default=30,
                        help="instances per configuration (denominator of 'Solved'); 0: those found")
    parser.add_argument("--time-decimals", type=int, default=2)
    parser.add_argument("--ratio-decimals", type=int, default=4)
    parser.add_argument("--no-bold", action="store_true")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--font-size", default="small",
                        help="small, footnotesize, scriptsize, tiny...")
    parser.add_argument("--tabcolsep", default=None, help="space between columns, e.g. 3pt")
    parser.add_argument("--short-headers", action="store_true")
    parser.add_argument("--one-line-names", action="store_true",
                        help="method names on one line in the wide layout")
    parser.add_argument("--landscape", action="store_true", help="sidewaystable (package rotating)")
    args = parser.parse_args()

    latex_comparison_table(args.results_dir, out_path=args.out, layout=args.layout,
                           metrics=args.metrics, methods=args.methods, w_types=args.w_types,
                           n_values=args.n, nb_instances=args.nb_instances or None,
                           time_decimals=args.time_decimals,
                           ratio_decimals=args.ratio_decimals, bold_best=not args.no_bold,
                           resize=args.resize, font_size=args.font_size, tabcolsep=args.tabcolsep,
                           short_headers=args.short_headers,
                           two_line_names=not args.one_line_names, landscape=args.landscape)
    print(f"LaTeX tables written to {args.out}")
