import Loading_Datasets as ld
import numpy as np


def sigmoid(input):
    output = 1 / (1 + np.exp(-input))
    return output


train_data_count = 200

w1 = np.random.normal(0, 1, (150, 102))
w2 = np.random.normal(0, 1, (60, 150))
w3 = np.random.normal(0, 1, (4, 60))

b1 = np.zeros((150, 1))
b2 = np.zeros((60, 1))
b3 = np.zeros((4, 1))

train_data_set = ld.train_set[:train_data_count]
out_set = np.zeros((train_data_count, 1))

for i in range(train_data_count):
    a0 = train_data_set[i][0]
    a1 = sigmoid((w1 @ a0) + b1)
    a2 = sigmoid((w2 @ a1) + b2)
    a3 = sigmoid((w3 @ a2) + b3)
    out_set[i] = np.argmax(a3)

corrects = 0
for i in range(train_data_count):
    if out_set[i] == np.argmax(train_data_set[i][1]):
        corrects += 1

accuracy = (corrects / train_data_count) * 100
print(accuracy)
