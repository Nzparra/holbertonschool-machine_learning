#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xticks(np.arange(0, 110, step=10))
plt.yticks(np.arange(0, 35, step=5))
plt.xlim(0, 100)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.hist(student_grades, edgecolor="black", bins=(np.arange(0, 110, 10)))
plt.show()
