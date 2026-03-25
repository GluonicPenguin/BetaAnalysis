import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

ufs = 24
dict_of_vars_xaxis = {'amplitude':'Amplitude [mV]', 'charge':'Charge [fC]', 'gain':'Gain'}

def main():
    if len(sys.argv) != 3:
        print("Usage: python langaus_curves_overlay.py <input_csv1_data> <input_csv2>")
        sys.exit(1)

    input_file = sys.argv[1]
    input_file_ltfs = sys.argv[2]
    df = pd.read_csv(input_file)
    df_ltfs = pd.read_csv(input_file_ltfs)
    possible_vars = ["amplitude", "charge", "gain"]
    available_vars = [v for v in possible_vars if v in df.columns]
    renamed_vars = ["A","Q","G"]

    cols = df_ltfs.columns[df_ltfs.columns.str.contains("_LTF") & ~df_ltfs.columns.str.endswith("_LTF_max")]
    df_ltfs_only = df_ltfs[cols]

    if not available_vars:
        raise ValueError("No amplitude, charge, or gain columns found in the CSV.")

    for variter, var in enumerate(available_vars):
        x_col = var
        y_cols = [c for c in df.columns if c.startswith(f"{var}_EVENTS_")]
        if not y_cols:
            continue

        ltf_headers_this_var = renamed_vars[variter] + "_LTF"
        ltfs_this_var = df_ltfs_only[ltf_headers_this_var]
        fig, ax = plt.subplots(figsize=(16, 10))
        markers = ['o','s','^','v','D','p','h','*','P','X','8']
        colours = ['r','g','b','orange','purple','brown','navy','lime','darkred','gold','magenta']
        iter = 0
        for y_col in y_cols:
            bias = y_col.split("_")[-1]
            x = df[x_col]
            y = df[y_col]

            y_norm = y / y.max() if y.max() != 0 else y
            '''
            ax.scatter(
                x,
                y_norm,
                marker=markers[iter],
                s=20,
                facecolors=to_rgba(colours[iter], 0.8),
                edgecolors=colours[iter],
                linewidths=1.5,
                label=bias+", LTF = "+str(round(ltfs_this_var[iter],3))
            )
            '''
            ax.plot(
                x,
                y_norm,
                color=colours[iter],
                linewidth=4,
                label=bias + ", LTF = " + str(round(ltfs_this_var[iter], 3))
            )
            iter = iter+1

        ax.set_xlabel(dict_of_vars_xaxis[x_col], fontsize=ufs)
        ax.set_ylabel("Normalised events", fontsize=ufs)
        ax.set_ylim(0, 1.06)
        ax.set_xlim(xmin=0)
        ax.tick_params(axis='both', labelsize=ufs)
        ax.legend(markerscale=3.5, frameon=True, fontsize=ufs)
        ax.grid(True)
        output_file = f"{os.path.splitext(input_file)[0]}_{var}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    main()
