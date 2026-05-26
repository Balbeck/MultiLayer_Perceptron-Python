from datasetManipulationClass import DatasetManipulation

CSV_DATA_FILEPATH = './data/data.csv'



def main():

	df = DatasetManipulation()
	df.read_csv(filepath=CSV_DATA_FILEPATH, has_header=False)
	df.drop_columns(columns=[0]) # column ID Number
	df.get_unique_values(column=0) # uniques values column Diagnosis (M = malignant, B = benign) 
	df.encode_column_values(column=0, mapping={'M': 1.0, 'B':0.0})
	df.create_train_and_test_dataset(train_ratio=0.7)

	return



if __name__ == "__main__":
	main()
