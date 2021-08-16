import tensorflow as tf
import numpy as np
from random import random
from sklearn.model_selection import train_test_split


def generate_dataset(num_samples, test_size=0.3):
    """Generate train/test data for sum operations.

    Args:
        num_samples (int): Number of total samples in dataset
        test_size (float, optional): Ratio of  num_samples used as test set. Defaults to 0.3.

    Returns:
        x_train (ndarray): 2d array with input data for training
        x_test (ndarray):  2d array with input data for testing
        y_train (ndarray): 2d array with target data for training
        y_test (ndarray):  2d array with target data for testing
    """
    # build inputs and targets
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0]+i[1]] for i in x])

    # split into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # create dataset with 500 samples
    x_train, x_test, y_train, y_test = generate_dataset(10000, 0.3)

    # build model with 3 layers: 2 -> 5 -> 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # choose optimiser
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.05)

    # compile model
    model.compile(optimizer=optimiser, loss="mse")

    # train model
    model.fit(x_train, y_train, epochs=100)

    # evaluate model on test set
    print("\nEvaluation on the test set:")
    model.evaluate(x_test, y_test, verbose=1)

    # get predictions
    data = np.array([[0.3, 0.2], [0.23, 0.1]])
    predictions = model.predict(data)

    print("\nPredictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))
