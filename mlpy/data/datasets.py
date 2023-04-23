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
from prettytable import PrettyTable

# native
import re
from copy import deepcopy


class Dataset:
    """Loader for csv datasets
    """
    def __init__(self, path: str):
        """

        Args:
            path (str): Path to dataset
        """
        self.data = self.__parse_csv(path)
     
    def __str__(self) -> str:
        table = PrettyTable()
        for column in self.header:
            table.add_column(column, self.data[column])
        return table.get_string()
    
    def __getitem__(self, item):
         return self.data[item]
     
    def __setitem__(self, name, item):
         self.data[name] = item
         
    def sort_by_column(self, column_name: str):
        """Sorts values in the data based on the values in a given column.
        Operation is performed in-place.

        Args:
            column_name (str): Column name
        """
        idxs = np.argsort(self[column_name])
        for column in self.header:
            self.data[column] = list(np.array(self.data[column])[idxs])
            
    def head(self, number_of_lines: int):
        head = deepcopy(self)
        for column in head.header:
            head[column] = head[column][:number_of_lines]
        return head
       
    def __parse_csv(self, path: str) -> dict:
        """Parse csv file

        Args:
            path (str): Path to csv file

        Returns:
            dict: Dictionary containing data
        """
        file = open(path)
        data = dict()
        header = [i.strip("\"") for i in file.readline().strip("\n").split(",")]
        for i in header:
            data[i] = []
        self.header = header
        while True:
            line = file.readline()
            if line == "":
                break
            line = self.__parse_line(line)
            for i, column in enumerate(header):
                data[column].append(line[i])
        return data
            
    def __parse_line(self, line: str) -> list:
        """Parses a line from csv file converting strings into the 
        appropriate format

        Args:
            line (str): Line of csv file

        Returns:
            list: parsed line
        """
        rfloat = re.compile("[-+]? (?: (?: \d* \. \d+ ) " \
                    + "| (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?", re.VERBOSE)
        line = [i for i in line.strip("\n").split(",")]
        line_parsed = []
        for substring in line:
            if bool(rfloat.match(substring)):
                if substring.isdigit():
                    value = int(substring)
                else:
                    value = float(substring)
            elif substring in ["True", "False"]:
                value = int(eval(substring))
            else:
                value = substring
            line_parsed.append(value)
        return line_parsed