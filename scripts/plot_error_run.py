import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import numpy as np

plt.rc("font", family="serif", serif="Times New Roman")
plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")
plt.rc("text", usetex=True)
plt.rc("savefig", dpi=500)
plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")
plt.rcParams.update({"font.size": 11, "legend.fancybox": False})
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)

if __name__ == "__main__":
    outputs = glob.glob("run_*/output.json")
    keys = json.loads(open(outputs[0]).read()).keys()
    data_raw = dict(zip(keys, [[] for kx in keys]))
    for output in outputs:
        d = json.loads(open(output).read())
        for keyx in keys:
            data_raw[keyx].append(d[keyx])

    df = pd.DataFrame(data_raw)

    print(df.head())

    cmap = cm.viridis

    cell_counts = sorted(df["N_cells"].unique())
    for cell_count in cell_counts:

        df_ncell = df.loc[df["N_cells"] == cell_count]

        df_ncell = df_ncell.reset_index()

        p_values = sorted(df_ncell["p"].unique())

        W = 6
        H = 4
        
        for error_type in ("errornorm", "errornorm_cg"):

            fig, ax = plt.subplots(1, 1, figsize=(W, H))

            norm_func = mpl.colors.LogNorm(vmin=1, vmax=max(p_values))

            for px in reversed(p_values):

                df_p = df_ncell.loc[df_ncell["p"] == px].reset_index()
                df_p = df_p.sort_values("N_particles")

                x = df_p.N_particles.to_numpy()
                y = df_p[error_type].to_numpy()

                lin_ab = np.polyfit(np.log10(x), np.log10(y), 1)

                if cell_count == 16:
                    # import pdb; pdb.set_trace()
                    pass

                ax.plot(
                    (x[0], x[-1]),
                    (
                        (10.0 ** lin_ab[1]) * x[0] ** lin_ab[0],
                        (10.0 ** lin_ab[1]) * x[-1] ** lin_ab[0],
                    ),
                    linestyle=":",
                    color=cmap(norm_func(px)),
                )

                ax.plot(
                    x,
                    y,
                    label=r"$p={}$, ${: 5.2f} N ^ {{{: 5.2f}}}$".format(
                        px,
                        10.0 ** lin_ab[1],
                        lin_ab[0],
                    ),
                    color=cmap(norm_func(px)),
                )

            ax.legend()

            # ax.set_yscale('log')
            # ylim = 1.05 * df.time_per_step.max()
            # ax.set_ylim(0, ylim)

            # ax.add_patch(Rectangle((0.0, 0.0), 1, ylim, facecolor="lightgrey"))

            ax.set_xlabel("Particle Count $N$")
            ax.set_ylabel("L2 error norm")
            ax.set_xscale("log")
            ax.set_yscale("log")

            # ax.set_xticks(node_counts)
            # ax.set_xticklabels([format_node_count(cx) for cx in node_counts])

            # handles, labels = ax.get_legend_handles_labels()
            # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            # ax.legend(handles, labels)

            # ax2 = ax.twiny()
            # ax2.set_xscale('log')
            # ax2.set_xlim(ax.get_xlim())
            # ax2.set_xticks(node_counts)
            # ax2.set_xticklabels([prettyInt(npart_total / (cx * cores_per_node)) for cx in node_counts])
            # ax2.set_xlabel(rf"Particles per {plot_config.get('compute_unit', 'core')}")
            # ax2.tick_params(axis='x', which='minor', top=False, bottom=False)
            # ax2.minorticks_off()

            ax.tick_params(axis="x", which="minor", top=False, bottom=False)
            ax.minorticks_off()

            fig.savefig(f"{error_type}_run_p_ncells_{cell_count}.pdf", bbox_inches="tight")
