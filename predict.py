import numpy as np
import matplotlib.pyplot as plt
from DatasetManipulationClass import DatasetManipulation
from multiLayerPerceptron import MultiLayerPerceptron


CSV_TEST = './data/test_dataset_20260526_1735.csv'




def to_one_hot(y):
	one_hot = np.zeros((y.shape[0], 2))
	one_hot[np.arange(y.shape[0]), y.astype(int).flatten()] = 1
	return one_hot


def plot_confusion_matrix(cm):
	fig, ax = plt.subplots()
	im = ax.imshow(cm, cmap='Blues')
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(['Benign', 'Malignant'])
	ax.set_yticklabels(['Benign', 'Malignant'])
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Actual')
	ax.set_title('Confusion Matrix')
	for i in range(2):
		for j in range(2):
			ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=14)
	plt.colorbar(im)
	plt.tight_layout()



def main():
	test = DatasetManipulation()
	test.read_csv(CSV_TEST, has_header=True)

########################################################################
########################################################################
#								TESTS
########################################################################

	# corelationClairementFaible = ['3', '6', '10', '11', '13', '16', '17', '20', '21', '23', '26', '30', '31']
	corelationClairementFaible = ['3', '6', '10', '11', '13', '16', '17', '20']
	# test.drop_columns(corelationClairementFaible)

########################################################################
########################################################################

	X_test = test.df.drop(columns=['1']).values
	y_test = test.df['1'].values.reshape(-1, 1)
	y_test = to_one_hot(y_test)

	model = MultiLayerPerceptron()
	mean, std = model.load_model()
	# mean, std = model.load_model(filepath='./saved_model_Best.npy')
	X_test = (X_test - mean) / std

	predictions = model.predict(X_test)
	labels = np.argmax(y_test, axis=1)

	accuracy = np.mean(predictions == labels)
	print(f" 🎯 Accuracy:  {accuracy:.4f}")

	# [ Confusion Matrix ]
	cm = np.zeros((2, 2), dtype=int)
	for true, pred in zip(labels, predictions):
		cm[true][pred] += 1
	
	tp, fn, fp, tn = cm[1][1], cm[1][0], cm[0][1], cm[0][0]
	precision = tp / (tp + fp + 1e-15)
	recall = tp / (tp + fn + 1e-15)
	f1 = 2 * precision * recall / (precision + recall + 1e-15)

	print(f" ✅ Precision: {precision:.4f}")
	print(f" 🔁 Recall:    {recall:.4f}")
	print(f" 🏆 F1 Score:  {f1:.4f}")
	print(f"\n Confusion Matrix:\n{cm}")

	# plot_confusion_matrix(cm)
	# plt.show()



if __name__ == "__main__":
	main()
