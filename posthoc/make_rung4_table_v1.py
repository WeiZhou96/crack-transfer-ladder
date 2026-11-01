"""Supplementary table of the rung-4 joint ladder under alternative background and joint definitions.

Run after rung4_variants_v1.py.
usage: python -X utf8 make_rung4_table_v1.py [<rung4_variants.json>] <output dir>
"""
import json
import sys
from pathlib import Path

C = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 2 else C / "results/ladder/rung4_variants.json"
OUT = Path(sys.argv[-1])
V = json.load(open(SRC, encoding="utf-8"))["variants"]
NAME = {"B10": "CF-Unpaired", "B11": "CF-Paired", "A_x01only": "CF-NegOnly", "A_x1only": "CF-CrackLayers",
        "A_paired_noaux": "CF-Paired w/o pair losses", "A_x00only": "CF-BgOnly", "B12_auto_posthoc": "MinedNeg\\,+\\,CF",
        "B1": "StrongAug", "B1_v5": "PhotoAug", "B2": "RandNeg", "B3_auto": "MinedNeg", "B0_v6": "Real-only, retrained",
        "R50_B11": "CF-Paired (R50)", "R50_B2": "RandNeg (R50)"}
VS_REAL = ["B10", "B11", "A_x01only", "A_x1only", "A_paired_noaux", "A_x00only", "B12_auto_posthoc", "B1", "B1_v5",
           "B2", "B3_auto", "B0_v6", "R50_B11", "R50_B2"]
VS_BG = ["B10", "B11", "A_x01only", "A_x1only", "A_paired_noaux"]
ORDER = ["pooled", "same_image", "ejoint", "ejoint_same"]


def cell(st):
    r, (lo, hi) = st["ratio"], st["ci"]
    s = f"\\makecell{{{r:.2f}\\\\{{\\tiny[{lo:.2f}, {hi:.2f}]}}}}"
    return f"\\textcolor{{cigrey}}{{{s}}}" if lo <= 1 <= hi else s


def row(a, ref):
    cells = [cell(V[v][a][ref][k]) for v in ORDER for k in ("excess", "relative_risk")]
    return f"{NAME[a]} & " + " & ".join(cells) + " \\\\"


body = "\n".join(row(a, "R50_B0" if a.startswith("R50_") else "B0") for a in VS_REAL if a in V["pooled"])
body_bg = "\n".join(row(a, "A_x00only") for a in VS_BG if a in V["pooled"])
t = r"""\begin{table*}[!htbp]
\centering\scriptsize
\renewcommand{\arraystretch}{1.1}
\setlength{\tabcolsep}{2.5pt}
\caption{Rung 4 of the joint ladder under alternative definitions of the background and of the joint class (post hoc, native operating point): ratio of the joint-specific excess and of the relative share (RR) to the reference arm. Background: unassociated false alarms per Mpx of class-free area, pooled over all 5,665 label-negative dacl10k images or restricted to the images that contain the joint class. Joints: expansion joints and joint tape (1,006 images) or expansion joints only (403 images).}\label{tab:rung4-variants}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{@{}l *{8}{>{\centering\arraybackslash}X}@{}}
\toprule
 & \multicolumn{4}{c}{Expansion joints and joint tape} & \multicolumn{4}{c}{Expansion joints only} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}
 & \multicolumn{2}{c}{All images} & \multicolumn{2}{c}{Same images} & \multicolumn{2}{c}{All images} & \multicolumn{2}{c}{Same images} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
 & excess & RR & excess & RR & excess & RR & excess & RR \\
\midrule
\multicolumn{9}{@{}l}{\textit{Relative to Real-only of the same backbone}} \\
""" + body + r"""
\midrule
\multicolumn{9}{@{}l}{\textit{Relative to CF-BgOnly (ConvNeXt-T)}} \\
""" + body_bg + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}\scriptsize
\item ``All images'' is the definition used in the main text. 95\% paired image-bootstrap intervals (10,000 replicates); grey: interval includes 1.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""
(OUT / "tab_rung4-variants.tex").write_text(t, encoding="utf-8", newline="\n")
print("ok", OUT / "tab_rung4-variants.tex")
