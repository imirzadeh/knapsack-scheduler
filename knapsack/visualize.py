import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def draw_bar_plot(filename='./measures.csv'):
	df = pd.read_csv(filename)
	fig = plt.figure(figsize=(40, 10))  # Create matplotlib figure

	ax = fig.add_subplot(111)  # Create matplotlib axes
	ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

	width = 0.4
	# this is an illustration for Alala
	# x = [8, 6, 4, 2]

	pos = list(range(df.shape[0]))
	# BLUE = "#06c2ac"
	# RED = "#ff796c"
	RED = "#FD7272"
	BLUE =  "#3B3B98"
	colors = [BLUE, RED]

	fit_curves = {'energy_per_sample': 1, 'score': 2}
	for i, col in enumerate(fit_curves.keys()):
		wanted_ax = [ax, ax2][i]
		wanted_ax.bar([p + width * i for p in pos], df[col], width, color=colors[i])

	ax.set_ylim([0.0, 20.0])
	ax2.set_ylim([80, 100])

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


def static_bar_plot():
	models = [
		{'model': 'min energy', 'energy': 9840, 'score': 84.66},
		{'model': 'best score', 'energy': 16675, 'score': 96.33},
		{'model': 'dense\nneural net', 'energy': 14336, 'score': 91.95},
		{'model': 'sparse\nneural net', 'energy': 13554, 'score': 89.27},
		{'model': 'knapsack\nsolution(1)', 'energy': 10182, 'score': 85.47},
		{'model': 'knapsack\nsolution(2)', 'energy': 12915, 'score': 90.14},
		{'model': 'knapsack\nsolution(3)', 'energy': 13257, 'score': 90.57},
		{'model': 'knapsack\nsolution(4)', 'energy': 13599, 'score': 91.17},
		{'model': 'knapsack\nsolution(5)', 'energy': 15307, 'score': 94.07},
	]
	df = pd.DataFrame(models)
	df['energy'] = df['energy']/(df['energy'].max())*100
	print(df)

	sns.set_context("paper", rc={"lines.linewidth": 8,
								 'xtick.labelsize': 11,
								 'ytick.labelsize': 13,
								 'legend.fontsize': 11,
								 'axes.labelsize': 11})
	
	
	
	fig = plt.figure()  # Create matplotlib figure
	
	ax = fig.add_subplot(111)  # Create matplotlib axes
	ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.
	
	width = 0.25
	# x = [8, 6, 4, 2]
	
	pos = list(range(df.shape[0]))
	RED = "#FD7272"
	BLUE =  "#3B3B98"
	colors = [BLUE, RED]
	
	fit_curves = {'energy': 1, 'score': 2}
	for i, col in enumerate(fit_curves.keys()):
		wanted_ax = [ax, ax2][i]
		wanted_ax.bar([p + width * i for p in pos], df[col], width, color=colors[i])
	
	ax.set_ylim([55, 100])
	ax2.set_ylim([80, 100])
	
	ax.set_ylabel('Energy')
	ax.set_xlabel('Models')
	ax2.set_ylabel('Score')
	
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
	plt.tight_layout()
	plt.savefig('./result.png', dpi=200)

	
	
if __name__ == "__main__":
	static_bar_plot()
	# sns.set_context("talk", rc={"lines.linewidth": 4,
	# 							'xtick.labelsize': 10,
	# 							'ytick.labelsize': 20,
	# 							'legend.fontsize': 20,
	# 							'axes.labelsize': 20})
	# draw_bar_plot()
