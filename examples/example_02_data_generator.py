import matplotlib.pyplot as plt
from mlpy.data import data_generator

ldg = data_generator.LinearGenerator(100, 1, 5)
line_y = (ldg.X @ ldg.W).flatten()
X = ldg.X[:, 1:].flatten()
y = ldg.y.flatten()

figure = plt.figure()
plt.title("Linear data test")
plt.scatter(X, y, c="b", s=15)
plt.plot(X, line_y, c="r", label="Ground Truth line")
plt.legend()
plt.show()