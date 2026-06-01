import numpy as np
import matplotlib.pyplot as plt


class DataVisualization:

	def __init__(self, df):
		self.df = df # df Pandas
		self.data = None # data Numpy (numbers Only!)
		self.headers = None # column names
		self.n_sample = None # n lines
		self.n_features = None # n columns
		self.data_directory = './data'



	def display_dataset(self, label_column, n_cols=3, n_rows=3)-> None:
		"""
		Display all features as scatter plots, colored by label column.
		Pages of n_cols x n_rows plots, cycling through all features.
		Just a First General Overview of all datas ^^

		Usage:
			viz = DataVisualization(dm.df)
			viz.display_dataset(label_column='diagnosis')
		"""
		per_page   = n_cols * n_rows
		features   = [col for col in self.df.columns if col != label_column]
		labels     = self.df[label_column]
		n_features = len(features)
		n_pages    = int(np.ceil(n_features / per_page))

		# Colors per label value
		unique_labels = labels.unique()
		palette = {}
		colors_list = ['#E05C5C', '#4AABDB', '#6ECB8A', '#F5A623']
		for i, lbl in enumerate(sorted(unique_labels)):
			palette[lbl] = colors_list[i % len(colors_list)]

		for page in range(n_pages):
			start = page * per_page
			end   = min(start + per_page, n_features)
			page_features = features[start:end]
			n_plots = len(page_features)

			fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
			fig.suptitle(
				f'Dataset Overview — page {page + 1}/{n_pages}',
				fontsize=14, fontweight='bold', y=1.01
			)
			axes_flat = axes.flatten()

			for i, feat in enumerate(page_features):
				ax = axes_flat[i]
				for lbl in sorted(unique_labels):
					mask = labels == lbl
					ax.scatter(
						np.where(mask)[0],
						self.df.loc[mask, feat],
						c=palette[lbl],
						label=str(lbl),
						alpha=0.6,
						s=12,
						edgecolors='none'
					)
				ax.set_title(feat, fontsize=9, pad=4)
				ax.set_xlabel('sample index', fontsize=7)
				ax.tick_params(labelsize=7)
				ax.spines[['top', 'right']].set_visible(False)

			# Hide unused subplots
			for j in range(n_plots, per_page):
				axes_flat[j].set_visible(False)

			# Global legend (once per page)
			handles = [
				plt.Line2D([0], [0], marker='o', color='w',
							markerfacecolor=palette[lbl], markersize=8, label=str(lbl))
				for lbl in sorted(unique_labels)
			]
			fig.legend(handles=handles, loc='upper right', fontsize=9,
						frameon=True, title=label_column)

			plt.tight_layout()
			plt.show()
