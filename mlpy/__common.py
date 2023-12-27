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


class BaseModel:
    def __init__(self, name: str = "model"):
        self.name = name
        
    def fit(self, *args, **kwargs):
        raise NotImplementedError(f"Not available for {self.__class__}")
        
    def predict(self, *args, **kwargs):
        raise NotImplementedError(f"Not available for {self.__class__}")
    
    def predict_proba(self, *args, **kwargs):
        raise NotImplementedError(f"Not available for {self.__class__}")


class BaseTransform:
    def __init__(self, name: str = "transform"):
        self.name = name
        
    def fit(self, *args, **kwargs):
        raise NotImplementedError(f"Not available for {self.__class__}")
        
    def transform(self, *args, **kwargs):
        raise NotImplementedError(f"Not available for {self.__class__}")
    
    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError(f"Not available for {self.__class__}")