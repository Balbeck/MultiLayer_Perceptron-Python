import numpy as np
from DatasetManipulationClass import DatasetManipulation
from multiLayerPerceptron import MultiLayerPerceptron, Layer

import matplotlib.pyplot as plt


# CSV_TRAIN = './data/train_dataset_Full.csv'
# CSV_TRAIN = './data/train_dataset_Opti.csv'
CSV_TRAIN = './data/train_dataset_20260526_1735.csv'
CSV_TEST = './data/test_dataset_20260526_1735.csv'



def to_one_hot(y):
	one_hot = np.zeros((y.shape[0], 2))
	one_hot[np.arange(y.shape[0]), y.astype(int).flatten()] = 1
	return one_hot

def main():

	train = DatasetManipulation()
	test = DatasetManipulation()
	train.read_csv(CSV_TRAIN, has_header=True)
	test.read_csv(CSV_TEST, has_header=True)


########################################################################
########################################################################
#								TESTS
########################################################################
	'''
	Suite aux premieres Viz (DataVisualization.display_dataset(label_column=1))
	On remarque qu 'il y a une correlation qui se distingue
	entre features et Resultat 'B' ou 'M'
	'''
	# corelationForte = [2, 4, 5, 8, 9, 12, 14, 15, 18, 22, 24, 25, 28, 29]
	# corelationMoyenne = [7, 19, 27]
	# corelationClairementFaible = ['3', '6', '10', '11', '13', '16', '17', '20', '21', '23', '26', '30', '31']
	corelationClairementFaible = ['3', '6', '10', '11', '13', '16', '17', '20']
	# train.drop_columns(corelationClairementFaible)
	# test.drop_columns(corelationClairementFaible)






########################################################################
########################################################################

	# # # [ A reorganiser dans Classes !!! ]

	# [ Split X, y - data, result ]
	X_train = train.df.drop(columns=['1']).values
	y_train = train.df['1'].values.reshape(-1, 1)
	X_test = test.df.drop(columns=['1']).values
	y_test = test.df['1'].values.reshape(-1, 1)

	# [ Normalisation ] (std mean)
	mean = X_train.mean(axis=0)
	std = X_train.std(axis=0)
	X_train = (X_train - mean) / std
	X_test = (X_test - mean) / std

	# [ Encoding One-Hot ]
	y_train = to_one_hot(y_train)
	y_test = to_one_hot(y_test)

	model = MultiLayerPerceptron()

	# [ Best ]: 42 epochs 100% Accuracy !!! ^^^
	network = model.create_neural_network([
		Layer(X_train.shape[1], activation='relu'),
		Layer(21, activation='relu', weights_initializer='heUniform'),
		Layer(5, activation='relu', weights_initializer='heUniform'),
		Layer(2, activation='sigmoid', weights_initializer='heUniform')
	])	


	# network = model.create_neural_network([
	# 	Layer(X_train.shape[1], activation='sigmoid'),
	# 	Layer(42, activation='sigmoid', weights_initializer='heUniform'),
	# 	Layer(21, activation='sigmoid', weights_initializer='heUniform'),
	# 	Layer(5, activation='sigmoid', weights_initializer='heUniform'),
	# 	Layer(2, activation='sigmoid', weights_initializer='heUniform')
	# ])

	# # Best
	# network = model.create_neural_network([
	# 	Layer(X_train.shape[1], activation='relu'),
	# 	Layer(42, activation='relu', weights_initializer='heUniform'),
	# 	Layer(21, activation='relu', weights_initializer='heUniform'),
	# 	Layer(5, activation='relu', weights_initializer='heUniform'),
	# 	Layer(2, activation='sigmoid', weights_initializer='heUniform')
	# ])

	# network = model.create_neural_network([
	# 	Layer(X_train.shape[1], activation='sigmoid'),
	# 	Layer(12, activation='sigmoid', weights_initializer='heUniform'),
	# 	Layer(42, activation='relu', weights_initializer='heUniform'),
	# 	Layer(24, activation='relu', weights_initializer='heUniform'),
	# 	Layer(5, activation='sigmoid', weights_initializer='heUniform'),
	# 	Layer(2, activation='softmax', weights_initializer='heUniform')
	# ])



	# history = model.fit((X_train, y_train), (X_test, y_test), epochs=21)
	history = model.fit((X_train, y_train), (X_test, y_test), epochs=42)
	# history = model.fit((X_train, y_train), (X_test, y_test), epochs=126)
	# history = model.fit((X_train, y_train), (X_test, y_test), epochs=1000)

	# plt.plot(history['train_loss'], label='train loss')
	# plt.plot(history['val_loss'], label='val loss')
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.legend()
	# plt.show()

	model.save_model(mean=mean, std=std)


if __name__ == "__main__":
	main()
