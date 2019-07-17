import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def draw_bar_plot(filename='./measures.csv'):
	df = pd.read_csv(filename)
	fig = plt.figure(figsize=(80, 20))  # Create matplotlib figure

	ax = fig.add_subplot(111)  # Create matplotlib axes
	ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

	width = 0.4
	# x = [8, 6, 4, 2]

	pos = list(range(df.shape[0]))
	BLUE = "#06c2ac"
	RED = "#ff796c"
	colors = [BLUE, RED]

	fit_curves = {'energy_per_sample': 1, 'score': 2}
	for i, col in enumerate(fit_curves.keys()):
		wanted_ax = [ax, ax2][i]
		wanted_ax.bar([p + width * i for p in pos], df[col], width, color=colors[i])

	ax.set_ylim([0.0, 20.0])
	ax2.set_ylim([40, 100])

	ax2.set_ylabel('Accuracy')
	ax.set_ylabel('Energy(J)')
	ax.set_xlabel('Classifiers')

	# ax.legend(['student'], loc = 'upper center', bbox_to_anchor=(-0.24, 1.08, 1., .102))
	# ax2.legend(['teacher'], loc = 'upper center', bbox_to_anchor=(0.25, 1.08, 1., .102))

	plt.xticks([p + 0.15 * len(pos) * width for p in pos],
			   [row['model'] for i, row in df.iterrows()])

	ax2.spines['right'].set_color(RED)
	ax2.spines['left'].set_color(BLUE)

	ax.tick_params(axis='y', colors=BLUE)
	ax2.tick_params(axis='y', colors=RED)
	ax.yaxis.label.set_color(BLUE)
	ax2.yaxis.label.set_color(RED)
	plt.savefig('./result.png', dpi=200)

if __name__ == "__main__":
	sns.set_context("talk", rc={"lines.linewidth": 4,
								'xtick.labelsize': 10,
								'ytick.labelsize': 20,
								'legend.fontsize': 20,
								'axes.labelsize': 20})
	draw_bar_plot()
