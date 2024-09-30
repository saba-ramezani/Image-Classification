import Loading_Datasets as ld
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta


def sigmoid(input):
    output = 1 / (1 + np.exp(-input))
    return output


def sigmoid_derivation(input):
    output = sigmoid(input) * (1 - sigmoid(input))
    return output


def plot(costs):
    n = np.arange(0, number_of_epochs, 1)
    plt.plot(n, costs, 'r')
    plt.show()


train_data_count = 200
learning_rate = 1
number_of_epochs = 20
batch_size = 10
total_time = timedelta()
total_cost = 0
total_accuracy = 0


for r in range(10):
    print(f"Round : {r}")
    train_data_set = ld.train_set[:train_data_count]
    batch_count = int(np.floor(train_data_count / batch_size))
    costs = np.zeros(number_of_epochs)
    accuracies = np.zeros(number_of_epochs)

    w1 = np.random.normal(0, 1, (150, 102))
    w2 = np.random.normal(0, 1, (60, 150))
    w3 = np.random.normal(0, 1, (4, 60))

    b1 = np.zeros((150, 1))
    b2 = np.zeros((60, 1))
    b3 = np.zeros((4, 1))

    start_time = datetime.now()
    for e in range(number_of_epochs):
        np.random.shuffle(train_data_set)
        corrects = 0
        cost = 0
        for b in range(batch_count):

            batch = train_data_set[b * batch_size:(b + 1) * batch_size]

            grad_w1 = np.zeros((150, 102))
            grad_w2 = np.zeros((60, 150))
            grad_w3 = np.zeros((4, 60))

            grad_b1 = np.zeros((150, 1))
            grad_b2 = np.zeros((60, 1))
            grad_b3 = np.zeros((4, 1))

            for image in batch:
                a0 = image[0]
                z1 = (w1 @ a0) + b1
                a1 = sigmoid(z1)
                z2 = (w2 @ a1) + b2
                a2 = sigmoid(z2)
                z3 = (w3 @ a2) + b3
                a3 = sigmoid(z3)
                output = np.argmax(a3)

                if output == np.argmax(image[1]):
                    corrects += 1

                # grad_w3
                grad_w3 += (sigmoid_derivation(z3) * 2 * (a3 - image[1])) @ np.transpose(a2)

                # grad_a2
                grad_a2 = np.transpose(w3) @ (sigmoid_derivation(z3) * 2 * (a3 - image[1]))

                # grad_w2
                grad_w2 += (sigmoid_derivation(z2) * grad_a2) @ np.transpose(a1)

                # grad_a1
                grad_a1 = np.transpose(w2) @ (sigmoid_derivation(z2) * grad_a2)

                # grad_w1
                grad_w1 += (sigmoid_derivation(z1) * grad_a1) @ np.transpose(a0)

                # grad_b3
                grad_b3 += (sigmoid_derivation(z3) * 2 * (a3 - image[1]))

                # grad_b2
                grad_b2 += (sigmoid_derivation(z2) * grad_a2)

                # grad_b1
                grad_b1 += sigmoid_derivation(z1) * grad_a1

                cost += (np.transpose((a3 - image[1])) @ (a3 - image[1]))[0, 0]

            w1 = w1 - (learning_rate * (grad_w1 / batch_size))
            w2 = w2 - (learning_rate * (grad_w2 / batch_size))
            w3 = w3 - (learning_rate * (grad_w3 / batch_size))
            b1 = b1 - (learning_rate * (grad_b1 / batch_size))
            b2 = b2 - (learning_rate * (grad_b2 / batch_size))
            b3 = b3 - (learning_rate * (grad_b3 / batch_size))

        costs[e] = cost / train_data_count
        accuracies[e] = corrects / train_data_count * 100
        print(f"Epoch {e} : Accuracy = {accuracies[e]} , Cost = {costs[e]}")

    end_time = datetime.now()
    learning_time = end_time - start_time
    total_time += learning_time
    total_cost += costs[-1]
    total_accuracy += accuracies[-1]
    print(f"Round {r} Learning Time : {learning_time}")
    if r == 9:
        plot(costs)

print(f"Average Learning Time : {total_time/10}")
print(f"Average Cost : {total_cost/10}")
print(f"Average Accuracy : {total_accuracy/10}")

