# MNSIT-Neural-network

This repository contains a simple two-layer neural network for handwritten digit recognition using the MNIST dataset. The neural network is implemented using NumPy and trained with forward propagation, backpropagation, and the sigmoid activation function. The mean squared error (MSE) is used as the loss and cost function.

# Dependencies

- NumPy

- Matplotlib

- data.py containing the get_mnist() function to load the MNIST dataset

# How to run

- Ensure you have the required dependencies installed.

- Download or clone the repository.

- Ensure the data.py file with the get_mnist() function is in the same directory as the main script.

# Run the main script:

```
python MNSIT_Neural_Network_Scratch.py

```

The script will train the neural network for 3 epochs, displaying the accuracy for each epoch.

After training, you will be prompted to enter a number between 0 and 59,999. This corresponds to the index of an image in the MNIST dataset. The neural network will make a prediction for the digit in the image and display it using Matplotlib.

To exit the program, use keyboard interrupts (e.g., Ctrl+C).
