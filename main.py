import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class NeuralNetwork:
	def __init__(self, num_input_nodes=8, num_hidden_nodes=4, num_output_nodes=1, momentum=False):
		self.momentum = momentum
		self.num_input_nodes = num_input_nodes
		self.num_hidden_nodes = num_hidden_nodes
		self.num_output_nodes = num_output_nodes

		self.rmse = list()  # list that holds the root mean squared error for each row of data
		validation_rmse = list()

		self.predicted_results = list()  # list that holds the final output for each row of data
		self.data = pd.read_csv("files/train.csv")

		# create the hidden layer weights and add a bias weight
		self.hidden_nodes_weights = np.random.uniform(-2 / num_input_nodes, 2 / num_input_nodes,
													  (num_hidden_nodes, num_input_nodes + 1))

		self.hidden_layer = list()  # create a list to hold the output of the nodes in the hidden layer

		# create the output layer weights and add a bias weight
		self.output_nodes_weights = np.random.uniform(-2 / num_hidden_nodes, 2 / num_hidden_nodes,
													  (num_output_nodes, num_hidden_nodes + 1))

		self.prev_update_output = 0
		self.prev_update_hidden = 0

	def get_predicted_results_length(self):
		return f"number of predictands is: {len(self.predicted_results)}"

	def get_rmse(self):
		return self.rmse

	def forward_pass(self):
		array = list()
		for i in range(self.data.shape[0]):  # iterate through the rows of data
			# input_layer is a list of the cells for the row of data
			input_layer = self.data.values[i][:self.num_input_nodes]
			input_layer = np.insert(input_layer, 0, 1)  # add a bias
			# print(f"{input_layer=}")

			# mulitiply the weights by the input layer
			weighted_sums = np.dot(self.hidden_nodes_weights, input_layer)
			weighted_sums = np.insert(weighted_sums, 0, 1)  # add a bias

			# add the output of the activation function to the hidden layer
			# apply the activation function to the weighted sum
			self.hidden_layer.append(sigmoid(weighted_sums))

			# multiply the weights by the hidden layer to get the weighted sums
			# feed the hidden layer into the output layer
			weighted_sums = np.dot(self.output_nodes_weights, self.hidden_layer[-1])

			# apply the activation function to the weighted sum of the output layer
			final_output = sigmoid(weighted_sums)

			# add the final output to the network
			self.predicted_results.append(final_output)

			# calculate the root mean squared error
			error = np.sqrt(np.mean((self.data.values[i][-1] - final_output) ** 2))
			array.append(error)

		mean = np.mean(array)
		self.rmse.append(mean)
		return mean

	def back_propagation(self, learning_rate=0.01):
		def sigmoid_prime(x):
			return x * (1 - x)

		for i in range(self.data.shape[0]):
			output_layer_error = self.data.values[i][-1] - self.predicted_results[i]
			output_layer_delta = output_layer_error * sigmoid_prime(self.predicted_results[i])

			# Assuming self.hidden_layer[-1] includes the bias as the first element
			sigmoid_output_hidden = self.hidden_layer[-1][1:]  # Exclude the bias unit
			sigmoid_prime_hidden = sigmoid_prime(sigmoid_output_hidden)

			# Correctly propagate the error back to the hidden layer
			# This requires reshaping output_layer_delta if it's not already in a suitable shape (e.g., making it a 1D array with a single element)
			# Adjust this line if output_layer_delta is not a scalar or if dimensions mismatch
			hidden_layer_error = np.dot(self.output_nodes_weights[:, 1:].T, output_layer_delta.reshape(-1, 1)).flatten()

			# Apply element-wise multiplication for the sigmoid_prime_hidden
			hidden_layer_delta = hidden_layer_error * sigmoid_prime_hidden

			# print(f"{hidden_layer_delta=}")
			# print(f"{hidden_layer_delta.shape=}")

			# Update weights from hidden to output layer
			for j in range(self.num_hidden_nodes + 1):  # +1 for the bias weight
				# Update the weights for the output layer; assumes single output node
				self.output_nodes_weights[0, j] += learning_rate * output_layer_delta * self.hidden_layer[i][
					j] + 0.9 * self.prev_update_output
				if self.momentum:
					self.prev_update_output = learning_rate * output_layer_delta * self.hidden_layer[i][j]

			# Update weights from input to hidden layer
			input_layer = np.insert(self.data.values[i][:self.num_input_nodes], 0, 1)  # Insert bias to the input layer
			for h in range(self.num_hidden_nodes):
				for j in range(self.num_input_nodes + 1):  # +1 for the bias
					# Update the weights for the hidden layer
					self.hidden_nodes_weights[h, j] += learning_rate * hidden_layer_delta[h] * input_layer[
						j] + 0.9 * self.prev_update_hidden
					if self.momentum:
						self.prev_update_hidden = learning_rate * hidden_layer_delta[h] * input_layer[j]

	def train_network(self, num_epochs=1000):
		validation_RMSE_array = list()
		for i in range(num_epochs):
			if i % 100 == 0:
				print(f"Training for {i} epochs")
			self.predicted_results.clear()
			self.forward_pass()
			self.back_propagation()

			self.data = pd.read_csv("files/validate.csv")

			x = self.forward_pass()
			self.rmse.pop()
			validation_RMSE_array.append(x)

		# print(f"{validation_RMSE_array=}")
		# print(f"{self.rmse=}")

		# self.plot_rmse(validation_RMSE_array)

		return self.rmse

	def plot_rmse(self, validation_rmse_array):
		plt.plot(self.rmse, label="Train")
		plt.plot(validation_rmse_array, label="Validation")
		plt.xlabel("Epochs")
		plt.ylabel("Root Mean Squared Error")
		plt.title("Root Mean Squared Error vs. Epochs")
		plt.legend()
		plt.show()

	def test_network(self):
		# print(f"{self.hidden_nodes_weights=}")
		# print(f"{self.output_nodes_weights=}")
		self.data = pd.read_csv("files/test.csv")
		x = self.forward_pass()
		self.rmse.pop()
		print(f"{x=}")
	# self.plot_rmse()


if __name__ == "__main__":
	arr = list()
	# for i in range(4, 17):
	nn = NeuralNetwork(num_hidden_nodes=8, momentum=False)

	# Assume you have a method to train the network
	nn.train_network(num_epochs=1000)
	nn.test_network()
	arr.append(nn.get_rmse())

	for i in arr:
		plt.plot(i, label=f"Hidden Nodes: {arr.index(i) + 4}")
	plt.xlabel("Epochs")
	plt.ylabel("Root Mean Squared Error")
	plt.title("Root Mean Squared Error vs. Epochs")
	plt.legend()
	plt.show()

# for i in arr:
# 	final_rmse = i[-1]
# 	# plot the nodes against the final rmse
# 	plt.scatter(arr.index(i) + 4, final_rmse)
# 	plt.xlabel("Number of Hidden Nodes")
# 	plt.ylabel("Final RMSE")
# 	plt.title("Number of Hidden Nodes vs. Final RMSE")
# plt.show()
