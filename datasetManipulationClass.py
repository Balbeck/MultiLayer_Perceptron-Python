import pandas as pd
import numpy as np
import os

from datetime import datetime

""" 
	class DatasetManipulation:
		- read_csv(self, filepath: str, has_header: bool)-> pd.DataFrame:	
		- drop_columns(self, columns)-> None:
		- get_unique_values(self, column)-> list:
		- encode_column_values(self, column, mapping)-> None:
		- create_train_and_test_dataset(self, train_ratio=0.7)-> None:
"""

class DatasetManipulation:

	def __init__(self):
		self.df = None
		self.data = None
		self.headers = None
		self.n_sample = None
		self.n_features = None
		self.data_directory = './data'


	def read_csv(self, filepath: str, has_header: bool)-> pd.DataFrame: # Return Pandas DataFrame Format
		try:
			df = pd.read_csv(filepath, header=0 if has_header else None) 
			df.replace('', pd.NA, inplace=True)
			self.headers = list(df.columns)

			print(f" 📈 [ {filepath} ] loaded and '' empty spaces replaced by pd.NA:")
			print(f"\t- shape: {df.shape}")
			print(f"\t- self.headers: {self.headers}") 
			self.df = df
			return self.df

		except :
			print(f"❌ [ Error ] read csv file: \'{filepath}\'")
			raise ValueError
	


	def drop_columns(self, columns)-> None:
		# Multi Type ^^: 0, '42', [0, 42] (index, name, multi index) ^^
		columns_to_drop = []
		if isinstance(columns, (str, int)):
			columns = [columns]
		for col in columns:
			if isinstance(col, int): # if int, provide df.columns[i] in Pd by default: df['str']
				columns_to_drop.append(self.df.columns[col])
			else:
				columns_to_drop.append(col)
		
		self.df = self.df.drop(columns=columns_to_drop)
		self.headers = list(self.df.columns)
		print(f" 🗑  Dropped columns: {columns_to_drop}")
		print(f"\t- new shape: {self.df.shape}")
		print(f"\t- new headers: {self.headers}") 


	def get_unique_values(self, column)-> list:
		if isinstance(column, int):
			column = self.df.columns[column]
		unique_values = self.df[column].unique()
		print(f" 🔍 Unique values in '{column}':\n{unique_values}")
		return unique_values


	def encode_column_values(self, column, mapping)-> None:
		"""
		Multi Type ^^: 0, 'tumor' (index: int or str) ^^
		Usage:
			- df.encode_column('tumor', {'M': 1.0, 'B': 0.0})
			- df.encode_column(0, {'M': 1.0, 'B': 0.0})
		"""
		try:
			if isinstance(column, int):
				column: str = self.df.columns[column]
			self.df[column] = self.df[column].map(mapping)
			print(f" 🔄 Encoded column: '{column}' with mapping: {mapping}")	

		except:
			print(f"❌ [ Error ] encode_column_values: \t -column: {column}\n\t -mapping: {mapping}'")
			raise ValueError


	def create_train_and_test_dataset(self, train_ratio=0.7)-> None:
		try:
			if not os.path.exists(self.data_directory):
				os.makedirs(self.data_directory)

			train = self.df.sample(frac=train_ratio, random_state=42)
			test = self.df.drop(train.index)

			date = datetime.now().strftime('%Y%m%d_%H%M')
			train_path = f'{self.data_directory}/train_dataset_{date}.csv'
			test_path = f'{self.data_directory}/test_dataset_{date}.csv'

			train.to_csv(train_path, index=False)
			test.to_csv(test_path, index=False)

			print(f" 💾 Saved: {train_path} {train.shape}")
			print(f" 💾 Saved: {test_path} {test.shape}")

		except Exception as e:
			print(f"❌ Error: {e}")
			raise
