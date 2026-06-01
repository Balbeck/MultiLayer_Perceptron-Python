import numpy as np


class Layer:

	def __init__(self, n_neurons, activation, weights_initializer='random'):
		self.n_neurons: int = n_neurons
		self.activation: str = activation
		self.weights_initializer: str = weights_initializer
		self.weights = None
		self.bias = None




class MultiLayerPerceptron:

	def __init__(self):
		self.neural_network = None



	def	create_neural_network(self, layers: Layer):
		for i in range(1, len(layers)):
			n_inputs = layers[i-1].n_neurons
			if layers[i].weights_initializer == 'heUniform':
				layers[i].weights = np.random.randn(n_inputs, layers[i].n_neurons) * np.sqrt(2 / n_inputs)
			else:
				layers[i].weights = np.random.randn(n_inputs, layers[i].n_neurons) * 0.01
			layers[i].bias = np.zeros((1, layers[i].n_neurons))
		self.neural_network = layers
		
		return self.neural_network



	def feed_forward(self, X):
		current_input = X
		
		for layer in self.neural_network[1:]:
			z = np.dot(current_input, layer.weights) + layer.bias
			current_input = self.activate_neuron(z, layer.activation)
			layer.output = current_input
		
		return current_input



	def activate_neuron(self, z, activation):
		if activation == 'sigmoid':
			return 1 / (1 + np.exp(-z))

		elif activation == 'softmax':
			exp = np.exp(z - np.max(z, axis=1, keepdims=True))
			return exp / np.sum(exp, axis=1, keepdims=True)

		elif activation == 'relu':
			return np.maximum(0, z)



	def backpropagation(self, X, y):
		m = X.shape[0]
		delta = self.neural_network[-1].output - y
		
		for i in reversed(range(1, len(self.neural_network))):
			layer = self.neural_network[i]
			prev_output = self.neural_network[i - 1].output if i> 1 else X
			layer.d_weights = np.dot(prev_output.T, delta) / m
			layer.d_bias = np.sum(delta, axis=0, keepdims=True) / m
			if i > 1:
				delta = np.dot(delta, layer.weights.T) * self.activate_derivative(self.neural_network[i-1].output, self.neural_network[i-1].activation)



	def activate_derivative(self, output, activation):
		if activation == 'sigmoid':
			return output * (1 - output)

		elif activation == 'relu':
			return (output > 0).astype(float)

		elif activation == 'softmax':
			return output * (1 - output)



	def fit(self, data_train, data_valid, learning_rate=0.0314, batch_size=8, epochs=42):
		X_train, y_train = data_train
		X_valid, y_valid = data_valid
		history = {'train_loss': [], 'val_loss': []}
		
		print(f"🌳 [ Epochs ]: {epochs}")
		for epoch in range(1, epochs + 1):
			for i in range(0, X_train.shape[0], batch_size):
				X_batch = X_train[i:i+batch_size]
				y_batch = y_train[i:i+batch_size]
				self.feed_forward(X_batch)
				self.backpropagation(X_batch, y_batch)
				self.update_weights(learning_rate)
			train_loss = self.compute_loss(X_train, y_train)
			val_loss = self.compute_loss(X_valid, y_valid)
			history['train_loss'].append(train_loss)
			history['val_loss'].append(val_loss)
			print(f"\t[ {epoch:02d} /{epochs}] - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

		return history



	def update_weights(self, learning_rate):
		for layer in self.neural_network[1:]:
			layer.weights -= learning_rate * layer.d_weights
			layer.bias -= learning_rate * layer.d_bias



	def compute_loss(self, X, y):
		output = self.feed_forward(X)
		m = X.shape[0]
		loss = -np.sum(y * np.log(output + 1e-15)) / m

		return loss



	def predict(self, X):
		output = self.feed_forward(X)

		return np.argmax(output, axis=1)



	def save_model(self, filepath='./saved_model.npy', mean=None, std=None):
		model = []

		for layer in self.neural_network:
			model.append({
				'weights': layer.weights,
				'bias': layer.bias,
				'activation': layer.activation,
				'n_neurons': layer.n_neurons,
				'weights_initializer': layer.weights_initializer
			})

		np.save(filepath, {'layers': model, 'mean': mean, 'std': std})
		print(f" 💾 saving model '{filepath}'...")



	def load_model(self, filepath='./saved_model.npy'):
		saved = np.load(filepath, allow_pickle=True).item()
		self.neural_network = []

		for layer_data in saved['layers']:
			layer = Layer(layer_data['n_neurons'], layer_data['activation'], layer_data['weights_initializer'])
			layer.weights = layer_data['weights']
			layer.bias = layer_data['bias']
			self.neural_network.append(layer)

		print(f" 💽 loading model '{filepath}'...")
		return saved['mean'], saved['std']
