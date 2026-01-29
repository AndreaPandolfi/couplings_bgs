import os
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cd = os.path.join("paper","laplace")
output_dir = os.path.join(cd, "output")


def plot_meeting_times_laplace(df, type='natural'):

    # df.groupby(['I', 'S'])['time'].mean()

    n_params = df.I.apply(sum).unique()

    fig, ax = plt.subplots(figsize=(6,3))
    for s in df.S.unique():
        df_s = df[df['S'] == s]
        means = df_s.groupby('I')['time'].mean()
        if type == 'natural':
            ax.plot(n_params, means.values, label=f'S={s}', marker='o', linestyle='-')
        elif type == 'loglog':
            ax.loglog(n_params, means.values, label=f'S={s}', marker='o', linestyle='-')
        elif type == 'errorbar':
            stds = df_s.groupby('I')['time'].std()
            ax.errorbar(means.index.map(sum), means.values, yerr=stds.values, label=f'S={s}', capsize=5, marker='o', linestyle='-')
    ax.set_xlabel('Parameters number', fontsize=15)
    ax.set_ylabel('Meeting Time', fontsize=15)
    ax.set_xticks(n_params, labels=n_params)
    ax.grid()
    ax.legend()
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    return fig, ax


df_bal = pd.read_csv(os.path.join(output_dir, "results_laplace_balanced.csv"))
df_bal = df_bal[(df_bal['var_fixed'] == False) & (df_bal.S != 3)]
df_bal['I'] = df_bal['I'].apply(lambda x: tuple(map(int, eval(x))))
fig_bal, ax_bal = plot_meeting_times_laplace(df_bal, type='natural')
ax_bal.set_yticks([10, 30, 50])
ax_bal.set_yticklabels([10, 30, 50])
fig_bal.savefig(os.path.join(output_dir, "laplace_balanced_nat_scale.pdf"), bbox_inches="tight")

fig_bal, ax_bal = plot_meeting_times_laplace(df_bal, type='errorbar')
ax_bal.set_yticks([10, 30, 50])
ax_bal.set_yticklabels([10, 30, 50])
fig_bal.savefig(os.path.join(output_dir, "laplace_balanced_errorbar.pdf"), bbox_inches="tight")

fig_bal, ax_bal = plot_meeting_times_laplace(df_bal, type='loglog')
ax_bal.set_yticks([10, 15, 30, 60])
ax_bal.set_yticklabels([10, 15, 30, 60])
fig_bal.savefig(os.path.join(output_dir, "laplace_balanced_loglog.pdf"), bbox_inches="tight")


df_unbal = pd.read_csv(os.path.join(output_dir, "results_laplace_unbalanced.csv"))
df_unbal = df_unbal[df_unbal.S != 3]
df_unbal['I'] = df_unbal['I'].apply(lambda x: tuple(map(int, eval(x))))

fig_unbal, ax_unbal = plot_meeting_times_laplace(df_unbal)
ax_unbal.set_yticks([50, 100, 150, 200, 250, 300, 350])
ax_unbal.set_yticklabels([50, 100, 150, 200, 250, 300, 350])
fig_unbal.savefig(os.path.join(output_dir, "laplace_unbalanced.pdf"), bbox_inches="tight")

fig_unbal, ax_unbal = plot_meeting_times_laplace(df_unbal, type='errorbar')
ax_unbal.set_yticks([50, 100, 150, 200, 250, 300, 350])
ax_unbal.set_yticklabels([50, 100, 150, 200, 250, 300, 350])
fig_unbal.savefig(os.path.join(output_dir, "laplace_unbalanced_errorbar.pdf"), bbox_inches="tight")

fig_unbal, ax_unbal = plot_meeting_times_laplace(df_unbal, type='loglog')
ax_unbal.set_yticks([100, 300])
ax_unbal.set_yticklabels([100, 300])
fig_unbal.savefig(os.path.join(output_dir, "laplace_unbalanced_loglog.pdf"), bbox_inches="tight")