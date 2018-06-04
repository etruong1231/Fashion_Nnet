import numpy as np
import csv
import random

## created a neural network for Fashion MNIST

class FashionNeuralNet():

	def __init__(self,dataset):
		''' constructor to make the weights'''
		## Input Layer = 28x28 picture => 784 nodes
		## hidden Layer 1 = 16 nodes
		## Hidden Layer 2 = 14 nodes
		## Output Layer = 10 nodes ( 1 each for output label)

		''' output labels
			0: T-shirt/top
			1: Trouser
			2: Pullover
			3: Dress
			4: Coat
			5: Sandal
			6: Shirt
			7: Sneaker
			8: Bag
			9: Ankle boot'''

		#loads up the data from csv
		print("\n*****Setting Up Training Dataset*****")
		filename = 'fashion-mnist_train.csv'
		raw_data = open(filename, 'rt')
		reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
		x = list(reader)
		data = np.array(x)
		np.random.seed(2)
		#gets the inputs from the csv
		# divide to normalize the inputs
		self.inputs = np.array(data[1:,1:,]).astype(np.float)/1000
		#print(self.inputs.shape)
		#print(self.inputs[0])

		#gets the output of the csv data
		#change the array into a transpose matrix
		self.outputs = np.array(data[1:,0]).astype(np.float).reshape((60000,1))

		self.new_outputs = np.zeros((60000,10))
		for counter,x in enumerate(self.outputs):
			self.new_outputs[counter-1][x[0]] = 1
		self.outputs = self.new_outputs


		neurons = [784,16,14,10]
		# -1 to add a bias for each layer
		self.weights = [2 * np.random.random((neurons[x], neurons[x+1]))-1 for x in range(3)]
		print("\n*****Created Neural Network*****")

	# Activation function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self,x):
		return x * (1-x)

	#feedForaward the neural network
	def feedForward(self,inputs):
		'''feed the neural network forward with the data and combines the weight'''
		# first layer of the neural network will be the input layer of 784 neurons
		input_layer = inputs
		# need to do a dot product sum to each neuron from to the first layer to the next layer of weights
		# need to call sigmoid to activate the neuron
		hidden_layer1 = self.sigmoid(np.dot(input_layer,self.weights[0]))
		# do the same as the last layer with the next layer of weights
		hidden_layer2 = self.sigmoid(np.dot(hidden_layer1, self.weights[1]))
		# get the final outcome to the last layer
		output_layer = self.sigmoid(np.dot(hidden_layer2, self.weights[2]))
		return input_layer, hidden_layer1, hidden_layer2, output_layer

	#backPropagation helps readjust the weights of each layers
	def backPropagation(self,input_layer,hidden_layer1,hidden_layer2, output_layer,output_error,learning_rate):
		''' readjust the weights of each layer'''
		# gets how far off we were from the output layer to the previous
		output_delta = output_error * self.sigmoid_derivative(output_layer)
		# gets the amount of error contributed from this layer to the next
		hidden_error2 = output_delta.dot(self.weights[2].T)
		# we do this for each layer
		hidden_delta2 = hidden_error2 * self.sigmoid_derivative(hidden_layer2)

		hidden_error1 = hidden_delta2.dot(self.weights[1].T)

		hidden_delta1 = hidden_error1 * self.sigmoid_derivative(hidden_layer1)



		# now we have to readjust the layers to make the weights better
		self.weights[2] += hidden_layer2.T.dot(output_delta) * learning_rate
		self.weights[1] += hidden_layer1.T.dot(hidden_delta2) * learning_rate
		self.weights[0] += input_layer.T.dot(hidden_delta1) * learning_rate


	def train(self, learning_rate = 1, epochs = 50, early_stopping = False,minibatch_size = 200):
		'''trains the neural network and feedforward, backpropagation to improve the neural network'''
		# train the neuron network for a amount of epochs
		print("\n*****Starting to train the Neural Network*****\n")
		previous_err = 100
		stop_counter = 0
		counter = 0 
		# does a mini batch sdg 
		for count in range(0, self.inputs.shape[0], minibatch_size):
			for train_count in range(0,epochs):
				random.shuffle(self.inputs)
				inputs = self.inputs[count: count+minibatch_size]
				outputs = self.outputs[count: count+minibatch_size]
				# need to feed the neuron network forward
				input_layer, hidden_layer1, hidden_layer2, output_layer = self.feedForward(inputs)
				# gets the predicted 
				output_error = outputs - output_layer
				err_rate = np.mean(np.abs(output_error))
				# for seeing the error rate change
				if(train_count == 0):
					print("Epochs: "+str(counter+1)+"/50 Error = "+str(err_rate)+"%")
				# after we calculate the error we need to backpropagate to take a step into gradient descent
				self.backPropagation(input_layer,hidden_layer1,hidden_layer2, output_layer,output_error,learning_rate)
			if(early_stopping):
				if(stop_counter == 2):
					break;
				if(err_rate < previous_err):
					previous_err = err_rate
				else:
					stop_counter += 1
			counter += 1

	def predict(self,raw_input):
		'''using the neural network to make a prediction on the input data'''
		try:
			input_layer, hidden_layer1, hidden_layer2, output_layer = self.feedForward(raw_inputs)
			print(np.argmax(output_layer))
		except e:
			print("Input Error: Input format was incorrect!")

	def testing_predict_error(self,testing_data):
		''' using the test data to test the neural network accuracy'''
		try:
			# formats the inputs and outputs
			test_inputs = np.array(testing_data[1:,1:,]).astype(np.float)/1000
			test_outputs = np.array(testing_data[1:,0]).astype(np.float).reshape((10000,1))
			formated_output = np.zeros((60000,10))
			for counter,x in enumerate(test_outputs):
				formated_outputs[counter-1][x[0]] = 1
			# feeds the neural network with the inputs
			input_layer, hidden_layer1, hidden_layer2, output_layer = self.feedForward(test_inputs)
			# total error
			total_err = np.mean(np.abs(output_layer - formated_outputs))
			print("Total Error of Testing Dataset: "+ str(total_err)+"%")
		except e:
			print("Input Error: Input/Output format was incorrect!")


		


if __name__ == "__main__":

	fnn = FashionNeuralNet('fashion-mnist_train.csv')
	fnn.train()
	fnn.testing_predict_error('fashion-mnis_text.csv')

	
