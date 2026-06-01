from DatasetManipulationClass import DatasetManipulation

CSV_DATA_FILEPATH = './data/data.csv'



def main():

	df = DatasetManipulation()
	df.read_csv(filepath=CSV_DATA_FILEPATH, has_header=False)
	df.drop_columns(columns=[0]) # column ID Number
	df.get_unique_values(column=0) # uniques values column Diagnosis (M = malignant, B = benign) 
	df.encode_column_values(column=0, mapping={'M': 1.0, 'B':0.0})
	df.create_train_and_test_dataset(train_ratio=0.7) # 70% Train
	# df.create_train_and_test_dataset(train_ratio=0) # Full Test
	# df.create_train_and_test_dataset(train_ratio=1) # Full Train

	return



# # To Create A Dataset oriented on Malign Detection ! ^^
# import pandas as pd
# def main():

# 	df = DatasetManipulation()
# 	df.read_csv(filepath=CSV_DATA_FILEPATH, has_header=False)
# 	df.drop_columns(columns=[0]) # column ID Number
# 	print(df.df.head())
# 	print(df.df[1].value_counts())
# 	df_M = df.df[df.df[1] == 'M']
# 	df_B = df.df[df.df[1] == 'B']
# 	df_train_opti = pd.concat([df_M, df_B.sample(frac=0.5)])
# 	df_train_opti[1] = df_train_opti[1].map({'M': 1.0, 'B': 0.0})
# 	df_train_opti = df_train_opti.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle

# 	train_path = './data/train_dataset_Opti.csv'
# 	df_train_opti.to_csv(train_path, index=False)
# 	print(f" 💾 Saved: {train_path} {df_train_opti.shape}")
# 	return


if __name__ == "__main__":
	main()
