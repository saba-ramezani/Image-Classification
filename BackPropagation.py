import Loading_Datasets as ld
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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
number_of_epochs = 5
batch_size = 10
train_data_set = ld.train_set[:train_data_count]
batch_count = int(np.floor(train_data_count / batch_size))
costs = np.zeros(number_of_epochs)

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
            for i in range(len(grad_w3)):
                for j in range(len(grad_w3[0])):
                    grad_w3[i][j] += a2[j][0] * sigmoid_derivation(z3[i][0]) * 2 * (a3[i][0] - image[1][i])

            # grad_a2
            grad_a2 = np.zeros((60, 1))
            for i in range(len(grad_w3[0])):
                for j in range(len(grad_w3)):
                    grad_a2[i][0] += w3[j][i] * sigmoid_derivation(z3[j][0]) * 2 * (a3[j][0] - image[1][j])

            # grad_w2
            for i in range(len(grad_w2)):
                for j in range(len(grad_w2[0])):
                    grad_w2[i][j] += grad_a2[i][0] * sigmoid_derivation(z2[i][0]) * a1[j][0]

            # grad_a1
            grad_a1 = np.zeros((150, 1))
            for i in range(len(grad_w2[0])):
                for j in range(len(grad_w2)):
                    grad_a1[i][0] += w2[j][i] * sigmoid_derivation(z2[j][0]) * grad_a2[j][0]

            # grad_w1
            for i in range(len(grad_w1)):
                for j in range(len(grad_w1[0])):
                    grad_w1[i][j] += grad_a1[i][0] * sigmoid_derivation(z1[i][0]) * a0[j][0]

            # grad_b3
            for i in range(len(grad_b3)):
                grad_b3[i][0] += 2 * (a3[i][0] - image[1][i]) * sigmoid_derivation(z3[i][0])

            # grad_b2
            for i in range(len(grad_b2)):
                grad_b2[i][0] += grad_a2[i][0] * sigmoid_derivation(z2[i][0])

            # grad_b1
            for i in range(len(grad_b1)):
                grad_b1 += grad_a1[i][0] * sigmoid_derivation(z1[i][0])

            for i in range(len(a3)):
                cost += (a3[i][0] - image[1][i]) ** 2


        w1 = w1 - (learning_rate * (grad_w1 / batch_size))
        w2 = w2 - (learning_rate * (grad_w2 / batch_size))
        w3 = w3 - (learning_rate * (grad_w3 / batch_size))
        b1 = b1 - (learning_rate * (grad_b1 / batch_size))
        b2 = b2 - (learning_rate * (grad_b2 / batch_size))
        b3 = b3 - (learning_rate * (grad_b3 / batch_size))

    costs[e] = cost / train_data_count
    print(f"Epoch {e} : Accuracy = {corrects / train_data_count * 100} , Cost = {cost / train_data_count}")

end_time = datetime.now()
print(f"Total Learning Time : {end_time - start_time}")
plot(costs)
