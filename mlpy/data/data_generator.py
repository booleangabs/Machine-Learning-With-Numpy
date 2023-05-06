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

# site-packages
import numpy as np


class NormalParamMeta:
    """Single normal distribution parameter metadata
    """
    def __init__(self, mean: float = 0, std: float = 1):
        """

        Args:
            mean (float, optional): Mean. Defaults to 0.
            std (float, optional): Standard deviation. Defaults to 1.
        """
        self.mean = mean
        self.std = std


class LinearRegressionData:
    """Data Generator for testing linear regressors
    
    First column is always a dummy variable (column of 1's)
    """
    def __init__(self, 
                 n_samples: int, 
                 n_features: int,
                 noise_std: float = 1,
                 random_state: int = None,
                 val_range: list = [-25, 25],
                 weight_meta: NormalParamMeta = NormalParamMeta(),
                 use_bias: bool = True
                ):
        """

        Args:
            n_samples (int): Number of samples
            n_features (int): Number of features
            random_state (Union[int, float]): Random seed for sampling. 
                                                Defaults to None.
            val_range (list, optional): Range of predictor values.
                                            Defaults to [-25, 25].
            weight_meta (ParamMeta, optional): Weight metadata. 
                                                Defaults to ParamMeta().
            use_bias (bool, optional): Use bias. Defaults to True.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_std = noise_std
        self.random_state = random_state
        self.val_range = val_range
        self.weight_meta = weight_meta
        self.use_bias = use_bias
        
        self.__generator = np.random.default_rng(self.random_state)
        self.__generate()
        
    def __generate(self):
        self.__W = self.__generator.normal(
                    self.weight_meta.mean, 
                    self.weight_meta.std,
                    (self.n_features + 1, 1)
                 )
        self.__W[0] *= self.use_bias
        
    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns predictors matrix and targets vector (in this order)

        Returns:
            tuple[np.ndarray, np.ndarray]: Predictors matrix, target vector
        """
        X_shape = (self.n_samples, self.n_features)
        X = self.__generator.uniform(*self.val_range, X_shape)
        dummy = np.ones((self.n_samples, 1))
        self.__X = np.hstack([dummy, X])
        eps = self.__generator.normal(0, self.noise_std, (self.n_samples, 1))
        self.__y = self.__X @ self.__W + eps
        return self.__X, self.__y
    
    def get_weights(self) -> np.ndarray:
        """Returns ground truth weights

        Returns:
            np.ndarray: Weights
        """
        return self.__W
    
