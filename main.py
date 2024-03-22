import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class NeuralNetwork:
	def __init__(self, num_input_nodes=8, num_hidden_nodes=4, num_output_nodes=1, learning_rate=0.01, momentum=False,
				 bold_driver=False):

		self.original_learning_rate = learning_rate
		self.original_num_hidden_nodes = num_hidden_nodes
		self.original_momentum = momentum
		self.original_bold_driver = bold_driver

		self.momentum = momentum
		self.bold_driver = bold_driver

		self.learning_rate = learning_rate  # learning rate
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
		self.final_activation = 0

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
			self.final_activation = sigmoid(weighted_sums)

			# add the final output to the network
			self.predicted_results.append(self.final_activation)

			# calculate the root mean squared error
			error = np.sqrt(np.mean((self.data.values[i][-1] - self.final_activation) ** 2))
			array.append(error)

		mean = np.mean(array)
		self.rmse.append(mean)
		return mean

	def back_propagation(self):
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
				# self.output_nodes_weights[0, j] += self.learning_rate * output_layer_delta * self.hidden_layer[i][j] + 0.9 * self.prev_update_output
				self.output_nodes_weights[0, j] += self.learning_rate * output_layer_delta * self.final_activation + 0.9 * self.prev_update_output
				if self.momentum:
					self.prev_update_output = self.learning_rate * output_layer_delta * self.final_activation

			# Update weights from input to hidden layer
			input_layer = np.insert(self.data.values[i][:self.num_input_nodes], 0, 1)  # Insert bias to the input layer
			sig_ws = sigmoid(np.dot(self.hidden_nodes_weights, input_layer))
			for h in range(self.num_hidden_nodes):
				for j in range(self.num_input_nodes):  # +1 for the bias
					# Update the weights for the hidden layer
					# print(f"{h=}, {j=}, {sig_ws=}")
					self.hidden_nodes_weights[h, j] += self.learning_rate * hidden_layer_delta[h] * sig_ws[h] + 0.9 * self.prev_update_hidden
					if self.momentum:
						self.prev_update_hidden = self.learning_rate * hidden_layer_delta[h] * sig_ws[h]

	def train_network(self, num_epochs=1000):
		validation_RMSE_array = list()
		print(f'{self.data.head()=}')
		print(f'{pd.read_csv("files/validate.csv").head()=}')
		for i in range(num_epochs):
			self.predicted_results.clear()
			self.forward_pass()
			self.back_propagation()

			self.data = pd.read_csv("files/validate.csv")

			x = self.forward_pass()
			self.rmse.pop()
			validation_RMSE_array.append(x)

			# update the learning rate
			if self.bold_driver:
				if i % 1 == 0 and i != 0:
					if validation_RMSE_array[-1] > validation_RMSE_array[-2]:
						self.learning_rate *= 0.7
					else:
						self.learning_rate *= 1.05

					# if the learning rate is too small, set it to 0.01
					if self.learning_rate < 0.001:
						self.learning_rate = 0.001
					elif self.learning_rate > 0.2:  # if the learning rate is too large, set it to 0.15
						self.learning_rate = 0.2

			if i % 1 == 0:
				print(f"Training for {i} epochs, lr = {self.learning_rate},training MSE = {self.rmse[-1]},validation MSE = {validation_RMSE_array[-1]}")

		# print(f"{validation_RMSE_array=}")
		# print(f"{self.rmse=}")

		self.plot_rmse(validation_RMSE_array)
		return self.rmse

	def test_network(self):
		# print(f"{self.hidden_nodes_weights=}")
		# print(f"{self.output_nodes_weights=}")
		self.data = pd.read_csv("files/test.csv")
		self.predicted_results.clear()
		x = self.forward_pass()
		self.rmse.pop()
		print(f"{x=}")
		self.plot_predicted_results()

	# self.plot_rmse()

	def plot_rmse(self, validation_rmse_array):
		global graph_counter
		plt.plot(self.rmse, label="Train")
		plt.plot(validation_rmse_array, label="Validation")
		plt.xlabel("Epochs")
		plt.ylabel("Root Mean Squared Error")
		plt.title(f"Root Mean Squared Error vs. Epochs, \n{self.original_momentum=}, {self.original_bold_driver=}, \n{self.original_learning_rate=}, {self.original_num_hidden_nodes=}")
		plt.plot(self.rmse.index(min(self.rmse)), min(self.rmse), "ro", label="Minimum RMSE")
		plt.legend()
		plt.savefig(os.path.join(graphs_dir, f"graph{graph_counter}.png"))
		graph_counter += 1
		plt.close()
		# plt.show()

	def plot_predicted_results(self):
		global graph_counter
		actual_results = [row[-1] for row in self.data.values]
		plt.scatter(self.predicted_results, actual_results)
		plt.xlabel("Predicted Result")
		plt.ylabel("Actual Result")
		plt.title(f"Predicted Results vs. Actual Results, \n{self.original_momentum=}, {self.original_bold_driver=}, \n{self.original_learning_rate=}, {self.original_num_hidden_nodes=}")

		# Calculate and plot the line of best fit
		slope, intercept = np.polyfit(actual_results, self.predicted_results, 1)
		best_fit_line = np.array(actual_results) * slope + intercept
		plt.plot(actual_results, best_fit_line, color='red', label='Line of Best Fit')

		# Plot the line y = x to show where perfect predictions would lie
		plt.plot([min(actual_results), max(actual_results)], [min(actual_results), max(actual_results)], color='black',
					linestyle='--', label='Perfect Predictions')

		plt.legend()
		plt.savefig(os.path.join(graphs_dir, f"graph{graph_counter}.png"))
		graph_counter += 1
		plt.close()
		# plt.show()


if __name__ == "__main__":
	matplotlib.use('TkAgg')
	plt.rcParams['figure.figsize'] = [8, 6]

	settings = [
		{'momentum': False, 'bold_driver': False},
		{'momentum': True, 'bold_driver': False},
		{'momentum': False, 'bold_driver': True},
		{'momentum': True, 'bold_driver': True}
	]

	# Store RMSE values for each setting and number of hidden nodes
	results = {setting_index: [] for setting_index in range(len(settings))}

	t_start = time.perf_counter()

	for num_hidden_nodes in range(4, 17):
		for setting_index, setting in enumerate(settings):
			nn = NeuralNetwork(num_hidden_nodes=num_hidden_nodes,
							   momentum=setting["momentum"],
							   bold_driver=setting["bold_driver"],
							   learning_rate=0.1)

			graph_counter = 0
			graphs_dir = f"pics/{nn.original_num_hidden_nodes=}-{nn.original_momentum=}-{nn.original_bold_driver=}"
			if not os.path.exists(graphs_dir):
				os.makedirs(graphs_dir)

			nn.train_network(num_epochs=10)
			nn.test_network()

			# Append RMSE of the final model to the results
			results[setting_index].append(nn.get_rmse()[-1])  # Assuming get_rmse() returns a list of RMSE over epochs

	# Plotting the results for each setting
	for setting_index, rmse_values in results.items():
		setting = settings[setting_index]
		plt.figure()
		plt.plot(range(4, 17), rmse_values, marker='o', linestyle='-')
		plt.xlabel("Number of Hidden Nodes")
		plt.ylabel("Root Mean Squared Error")
		plt.title(f"RMSE vs. Hidden Nodes\nMomentum={setting['momentum']}, Bold Driver={setting['bold_driver']}")
		plt.grid(True)
		plt.savefig(
			f"pics/setting_{setting_index + 1}_momentum_{setting['momentum']}_boldDriver_{setting['bold_driver']}.png")
		plt.close()

	t_end = time.perf_counter()
	print(f"Time taken: {t_end - t_start} seconds")
