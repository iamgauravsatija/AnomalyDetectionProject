from src.GuassainAlgorithm import GuassianAlgorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "data/ex8data1.xlsx" # change file name here

def main():
    plot_graph()
    guassianAlgorithm = GuassianAlgorithm(file_path)
    guassianAlgorithm.guassian_algorithm()

def plot_graph():
    file = pd.read_excel(file_path, sheet_name='X', header=None)
    plt.figure()
    plt.scatter(file[0], file[1])
    plt.show()


if __name__ == "__main__":
    main()