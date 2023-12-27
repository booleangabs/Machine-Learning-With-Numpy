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
from mlpy.data import data_generator as dg


datagen = dg.LinearRegressionData(100, 1, 5)
X, y = datagen.get_data()
weights = datagen.get_weights()

y_hat = (X @ weights).flatten()
X = X[:, 1:].flatten()
y = y.flatten()

figure = plt.figure()
plt.title("Linear data test")
plt.scatter(X, y, c="b", s=15)
plt.plot(X, y_hat, c="r", label="Ground Truth line")
plt.legend()
plt.show()