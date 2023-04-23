from mlpy.data import datasets

ds = datasets.Dataset("data_input/california_housing_test.csv")
print(ds.header)
print(ds.head(4))
