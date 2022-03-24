import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from math_utils import Math


class GramSchmidt():
    def __init__(self) -> None:
        self.a: DataFrame
        self.q: DataFrame
        self.r: DataFrame

        self.x: Series
        self.b: Series

        self.data: DataFrame
        self.filename: str
        self.sep: str


    def __init__(self, filename: str, sep: str) -> None:
        self.filename = filename
        self.sep = sep

# ---------------- Handling Matrices Q & R --------------------------- 
    def extract_data(self):
        self.data = pd.read_csv(self.filename, header=None, sep=self.sep)

    
    def configure_matrices(self):
        self.a = self.data.iloc[:, 0: self.data.shape[1] -1]
        self.b = self.data[self.data.shape[1] -1]


    def build_q_matrix(self) -> None:
        e = []
        for i in self.a.columns:
            ui = self.a[i]
            
            k = i
            while k >= 1:
                scalar_product_result = Math.scalar_product(self.a[k], e[k -1])
                tmp = Math.mult_vect_by(e[k -1], scalar_product_result)
                ui = ui.subtract(tmp)
                k-= 1            

            e.append(Math.mult_vect_by(ui, 1/Math.norm(ui)))

        self.q = pd.concat(e, axis=1, ignore_index=True)


    def build_r_matrix(self) -> None:
        self.r = pd.DataFrame(np.zeros((self.a.shape[1], self.a.shape[1])))
        for i in range(0, self.a.shape[1]):
            for j in range(i, self.a.shape[1]):
                self.r.iloc[i,j] = Math.scalar_product(self.a[j], self.q[i])

            for j in range(1, i):
                self.r.iloc[i,j] = 0

# ------------------------------------------------------------------------------------

# ---------------- Solving the System --------------------------- 
    def solve_system(self):
        # Calculate the transpose of the matrix Q
        qt = self.q.T

        # Performing the multiplication between the QT anb the b vector
        new_output = Math.matrix_multiplication(qt, self.b)
        print('The new output:')
        print(new_output)

        # Generate expressions and solve the system (to be continued ...)

        # for test ....
        # print()
        # print("The original Q matrix: ")
        # print(self.q)

        # print()
        # print("The QT matrix:")
        # print(qt)

        # print()
        # print("We must get the identity matrix: ")
        # resultant  = Math.matrix_multiplication(qt, self.q)
        # print(resultant.round(decimals=3))
