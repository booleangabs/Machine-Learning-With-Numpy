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
from mlpy.regression import KnnRegressor
from mlpy.metrics import mse

datagen = dg.LinearRegressionData(1000, 1, 5, random_state=1)
X, y = datagen.get_data()

lin_reg_pinv = LinearRegressor(name="pinv-lin-reg")
lin_reg_pinv.fit(X, y)

knn_reg = KnnRegressor(k=2)
knn_reg.fit(X, y)

colors = ["r", "g"]

ys = [
    model.predict(X).flatten() for model in [lin_reg_pinv, knn_reg]
]

errors = [
    mse(y.flatten(), y_hat) for y_hat in ys
]

labels_ = ["LinReg", "KnnReg"]
labels = [
    f"{label} - {error}" for label, error in zip(labels_, errors)
]

X = X[:, 1:].flatten()

figure = plt.figure()
plt.title("Regressors Test")
plt.scatter(X, y, c="pink", s=15)
plt.plot(X, ys[0], c=colors[0], label=labels[0])
plt.plot(np.sort(X), np.sort(ys[1]), c=colors[1], label=labels[1])
plt.legend()
plt.show()