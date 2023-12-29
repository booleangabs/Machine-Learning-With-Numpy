"""
MIT License

Copyright (c) 2023 Gabriel Tavares (booleangabs)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import matplotlib.pyplot as plt
import numpy as np
from mlpy.data import data_generator as dg
from mlpy.regression import LinearRegressor
from mlpy.metrics import mse

datagen = dg.LinearRegressionData(100, 1, 5, random_state=1)
X, y = datagen.get_data()

lin_reg_pinv = LinearRegressor(name="pinv-lin-reg")
lin_reg_pinv.fit(X, y)

lin_reg_gd = LinearRegressor(solver="gradient_descent", name="gd-lin-reg")
lin_reg_gd.fit(X, y)

lin_reg_sgd = LinearRegressor(solver="sgd", name="sgd-lin-reg")
lin_reg_sgd.fit(X, y)


colors = ["r", "g", "b"]

ys = [
    model.predict(X).flatten() for model in [lin_reg_pinv, lin_reg_gd, lin_reg_sgd]
]

errors = [
    mse(y.flatten(), y_hat) for y_hat in ys
]

labels_ = ["Pinv", "GD", "SGD"]
labels = [
    f"{label} - {error}" for label, error in zip(labels_, errors)
]

X = X[:, 1:].flatten()

figure = plt.figure()
plt.title("Linear Regression Test")
plt.scatter(X, y, c="pink", s=15)

for i in range(len(labels)):
    plt.plot(X, ys[i], c=colors[i], label=labels[i])
plt.legend()
plt.show()