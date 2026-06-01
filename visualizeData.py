from DatasetManipulationClass import DatasetManipulation
from DataVisualizationClass import DataVisualization


CSV_DATA_FILEPATH = './data/data.csv'



def main():

	dm = DatasetManipulation()
	dm.read_csv(CSV_DATA_FILEPATH, has_header=False)
	dm.drop_columns(columns=[0]) # remove ID column

	viz = DataVisualization(dm.df)
	viz.display_dataset(label_column=1)

	return



if __name__ == "__main__":
	main()
