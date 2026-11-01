# -*- coding: utf-8 -*-
"""Write two supplementary tables from the post hoc pre-submission analyses.

tab_indomain-matched.tex: PaveDistress false alarms against in-domain recall (from indomain_matched_v1.json and the
  arm curves of analyze_indomain_matched_v1.py).
tab_ladder-bgonly.tex: joint ladder ratios of the synthetic arms read against Real-only and against CF-BgOnly
  (point estimates; ratio of the two Real-only-referenced ratios, from review_reanalysis_add7.json).
Run after analyze_indomain_matched_v1.py.
usage: python -X utf8 make_supp_tables_v1.py <output dir>
"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
C = H.parent
OUT = Path(sys.argv[1])
sys.argv = sys.argv[:1]   # the analysis code executed below reads its default input
ns = {"__file__": str(H / "analyze_indomain_matched_v1.py")}
exec((H / "analyze_indomain_matched_v1.py").read_text(encoding="utf-8").split("out = {}")[0], ns)
curve, arms, rec, NAMES = ns["curve"], ns["arms"], ns["rec"], ns["NAMES"]
res = json.load(open(C / "results/posthoc/indomain_matched.json", encoding="utf-8"))["arms"]

ORDER = ["B0", "B0_v6", "B1", "B1_v5", "B2", "B2_g7", "B3_auto", "A_x00only", "A_x01only", "A_x1only", "A_paired_noaux",
         "B10", "B11", "B12_auto_posthoc", "A_x01only_B1", "B2_B1", "R50_B0", "R50_B2", "R50_B11"]
TEX = {"B12_auto_posthoc": r"MinedNeg\,+\,CF", "A_x01only_B1": r"PhotoAug\,+\,CF-NegOnly", "B2_B1": r"RandNeg\,+\,PhotoAug"}
rows = []
for a in ORDER:
    r = res[a]
    rc, fa = curve(arms[a], rec)
    name = TEX.get(a, NAMES[a])
    span = f"{rc.min():.3f}--{rc.max():.3f} & {fa[rc.argmin()]:.2f}--{fa[rc.argmax()]:.2f}"
    i = r.get("indomain")
    if i is None:
        tail = r"\multicolumn{3}{c}{reference}"
    elif i["ratio"] is None:
        n = i["seed_pairs_with_window"]
        tail = r"none & -- & " + (f"{round(i['frac_pairs_above_ref'] * n)}/{n}" if n else "--")
    else:
        tail = f"{i['interval'][0]:.3f}--{i['interval'][1]:.3f} & {i['ratio']:.2f} & " \
               f"{round(i['frac_pairs_above_ref'] * i['seed_pairs_with_window'])}/{i['seed_pairs_with_window']}"
    if a == "R50_B0":
        rows.append(r"\midrule")
    rows.append(f"{name} & {len(arms[a])} & {span} & {tail} \\\\")
t1 = r"""\begin{table*}[!htbp]
\centering\footnotesize
\renewcommand{\arraystretch}{1.1}
\caption{PaveDistress false alarms against in-domain recall (post hoc): threshold sweeps of the PaveDistress background test leg read against component recall on the 147 annotated PaveDistress frames instead of the RGB positive axis. Ratios follow Eq.~12 with in-domain recall; reference: Real-only of the same backbone.}\label{tab:indomain-matched}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{@{}l c c c c >{\centering\arraybackslash}X c@{}}
\toprule
Arm & Seeds & \makecell{In-domain recall\\($t=0.95$--$0.02$)} & \makecell{Density\\(components/Mpx)} & \makecell{Shared recall\\window} & \makecell{Matched-recall\\ratio} & \makecell{Seed pairs\\above} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}\scriptsize
\item Recall and density are seed means of the swept models at the highest ($0.95$) and lowest ($0.02$) threshold of the grid. ``none'': the arm-mean frontiers share no recall window, so no ratio is defined; seed pairs above: among the seed pairs whose frontiers share a window, the number in which the arm lies above the reference. Descriptive; no interval.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""
(OUT / "tab_indomain-matched.tex").write_text(t1, encoding="utf-8", newline="\n")

L = json.load(open(C / "results/ladder/review_reanalysis_add7.json", encoding="utf-8"))["ladder"]
r3, r4 = L["3_unseen_scene/joint"]["_density"], L["4_real/joint"]
ex3 = lambda a: r3[a]["x01"] - r3[a]["x00"]
BG = "A_x00only"
rows = []
for a in ["A_x01only", "A_x1only", "A_paired_noaux", "B10", "B11", "B12_auto_posthoc"]:
    v = [ex3(a) / ex3("B0"), ex3(a) / ex3(BG), r4[a]["excess"]["ratio"], r4[a]["excess"]["ratio"] / r4[BG]["excess"]["ratio"],
         r4[a]["relative_risk"]["ratio"], r4[a]["relative_risk"]["ratio"] / r4[BG]["relative_risk"]["ratio"]]
    rows.append(f"{TEX.get(a, NAMES[a])} & " + " & ".join(f"{x:.2f}" for x in v) + r" \\")
bg = [ex3(BG) / ex3("B0"), r4[BG]["excess"]["ratio"], r4[BG]["relative_risk"]["ratio"]]
t2 = r"""\begin{table}[!htbp]
\centering\footnotesize
\renewcommand{\arraystretch}{1.1}
\caption{Joint ladder of the synthetic arms read against Real-only and against CF-BgOnly (post hoc; ConvNeXt-T, point estimates; rung-4 intervals in Table~\ref{tab:rung4-variants}). Rung 3: distractor-specific ratio on rendered joints in unseen scenes. Rung 4: ratio of the joint-specific excess and of the relative share of joints over the class-free area of all dacl10k images.}\label{tab:ladder-bgonly}
\begin{tabularx}{\textwidth}{@{}l *{6}{>{\centering\arraybackslash}X}@{}}
\toprule
 & \multicolumn{2}{c}{Rung 3} & \multicolumn{2}{c}{Rung 4, excess} & \multicolumn{2}{c}{Rung 4, relative share} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Arm & vs Real-only & vs CF-BgOnly & vs Real-only & vs CF-BgOnly & vs Real-only & vs CF-BgOnly \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
""" + f"CF-BgOnly & {bg[0]:.2f} & -- & {bg[1]:.2f} & -- & {bg[2]:.2f} & -- \\\\" + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
(OUT / "tab_ladder-bgonly.tex").write_text(t2, encoding="utf-8", newline="\n")
print(t1[-1600:]); print(t2)
