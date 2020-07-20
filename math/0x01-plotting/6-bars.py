#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fruits = ['apples', 'bananas', 'oranges', 'peaches']
color = ['red', 'yellow', '#ff8000', '#ffe5b4']
size = np.arange(0, 3)
btt = np.array([0, 0, 0])
for i in range(len(fruit)):
    plt.bar(size, fruit[i], width=0.5, color=color[i], bottom=btt,
            label=fruits[i])
    btt = np.add(size, fruit[i])
names = ['Farrah', 'Fred', 'Felicia']
plt.xticks(size, names)
plt.yticks(np.arange(0, 90, 10))
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.legend()
plt.show()
