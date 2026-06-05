#!/usr/bin/env python3

import os
import ROOT

# ------------------------------------------------------------
# ROOT style
# ------------------------------------------------------------

ROOT.gStyle.SetOptStat(0)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

TREE_NAME = "Analysis"

NBINS_X = 500
XMIN = 0
XMAX = 50

NBINS_Y = 500
YMIN = 0
YMAX = 50

# ------------------------------------------------------------
# USER CONFIG
# ------------------------------------------------------------

# List of ROOT files
FILES = ["./stats_Sr_Run10_Ch1-170V_Ch2-165V_Ch3-2750V_trig230V.root",
"./stats_Sr_Run10_Ch1-175V_Ch2-170V_Ch3-2750V_trig230V.root",
"./stats_Sr_Run10_Ch1-180V_Ch2-175V_Ch3-2750V_trig230V.root",
"./stats_Sr_Run10_Ch1-185V_Ch2-180V_Ch3-2750V_trig230V.root"
]

# tmax[1] lower bounds
TMAX1_LOW = [-0.6,-0.6,-0.6,-0.6]

# tmax[1] upper bounds
TMAX1_HIGH = [1.2,1.2,1.2,1.1]

# tmax[2] lower bounds
TMAX2_LOW = [-0.6,-0.6,-0.6,-0.6]

# tmax[2] upper bounds
TMAX2_HIGH = [1.3,1.3,1.2,1.2]

# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

nfiles = len(FILES)

if not (
    len(TMAX1_LOW)  == nfiles and
    len(TMAX1_HIGH) == nfiles and
    len(TMAX2_LOW)  == nfiles and
    len(TMAX2_HIGH) == nfiles
):
    raise ValueError(
        "All tmax lists must have the same length as FILES"
    )

# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

OUTDIR_1 = "pmax_v_area_heatmaps_branch1"
OUTDIR_2 = "pmax_v_area_heatmaps_branch2"

os.makedirs(OUTDIR_1, exist_ok=True)
os.makedirs(OUTDIR_2, exist_ok=True)

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

for i, filename in enumerate(FILES):

    print(f"\nProcessing {filename}")

    if not os.path.exists(filename):
        print(f"File does not exist: {filename}")
        continue

    # Get cuts for this file
    t1_low  = TMAX1_LOW[i]
    t1_high = TMAX1_HIGH[i]

    t2_low  = TMAX2_LOW[i]
    t2_high = TMAX2_HIGH[i]

    # --------------------------------------------------------
    # Open ROOT file
    # --------------------------------------------------------

    f = ROOT.TFile.Open(filename)

    if not f or f.IsZombie():
        print(f"Could not open {filename}")
        continue

    tree = f.Get(TREE_NAME)

    if not tree:
        print(f"Tree '{TREE_NAME}' not found in {filename}")
        f.Close()
        continue

    base = os.path.splitext(os.path.basename(filename))[0]

    # ========================================================
    # Branch [1]
    # ========================================================

    c1 = ROOT.TCanvas(f"c1_{i}", "c1", 800, 700)

    hname1 = f"h1_{i}"

    h1 = ROOT.TH2F(
        hname1,
        f"{base}: pmax[1] vs area_new[1];area_new[1];pmax[1]",
        NBINS_X, XMIN, XMAX,
        NBINS_Y, YMIN, YMAX
    )

    draw_expr_1 = f"pmax[1]:area_new[1]>>{hname1}"

    cut_1 = (
        f"tmax[1] > {t1_low} && "
        f"tmax[1] < {t1_high}"
    )

    tree.Draw(draw_expr_1, cut_1, "COLZ")

    out1 = os.path.join(
        OUTDIR_1,
        f"{base}_branch1.png"
    )

    c1.SaveAs(out1)

    # ========================================================
    # Branch [2]
    # ========================================================

    c2 = ROOT.TCanvas(f"c2_{i}", "c2", 800, 700)

    hname2 = f"h2_{i}"

    h2 = ROOT.TH2F(
        hname2,
        f"{base}: pmax[2] vs area_new[2];area_new[2];pmax[2]",
        NBINS_X, XMIN, XMAX,
        NBINS_Y, YMIN, YMAX
    )

    draw_expr_2 = f"pmax[2]:area_new[2]>>{hname2}"

    cut_2 = (
        f"tmax[2] > {t2_low} && "
        f"tmax[2] < {t2_high}"
    )

    tree.Draw(draw_expr_2, cut_2, "COLZ")

    out2 = os.path.join(
        OUTDIR_2,
        f"{base}_branch2.png"
    )

    c2.SaveAs(out2)

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    f.Close()

# ------------------------------------------------------------
# Done
# ------------------------------------------------------------

print("\nDone.")
print(f"Branch [1] plots -> {OUTDIR_1}/")
print(f"Branch [2] plots -> {OUTDIR_2}/")
