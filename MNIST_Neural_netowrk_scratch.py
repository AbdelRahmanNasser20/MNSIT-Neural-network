from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset (images and labels)
images, labels = get_mnist()

# Initialize weights and biases for the input-to-hidden and hidden-to-output layers
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

# Set learning rate, number of correct predictions, and number of training epochs
learn_rate = 0.01
nr_correct = 0
epochs = 3

# Train the neural network for the specified number of epochs
for epoch in range(epochs):
    # Iterate through each image-label pair in the dataset
    for img, l in zip(images, labels):
        # Reshape the image and label arrays
        img.shape += (1,)
        l.shape += (1,)

        # Forward propagation from input to hidden layer
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))

        # Forward propagation from hidden to output layer
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Calculate the cost/error for the current prediction
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation: output layer to hidden layer
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # Backpropagation: hidden layer to input layer
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Display the accuracy for the current epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show the results for image indices
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    # Reshape the image array
    img.shape += (1,)

    # Forward propagation from input to hidden layer
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))

    # Forward propagation from hidden to output layer
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    # Display the image along with the predicted digit
    plt.title(f"Hire me if it's a {o.argmax()} :)")
    plt.show()