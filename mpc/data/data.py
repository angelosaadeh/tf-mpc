"""All the classes and functions that are related to importing data and that are common to all users"""
import pandas as pd
import pickle
import numpy as np


class Unpack:
    """The user's local directory"""
    def __init__(self, mod=2**60, precision=20):
        self.mod = mod
        self.precision = precision

    @staticmethod
    def get_file(file):
        """Imports a csv file as secret"""
        print('A file is being imported...')
        return pd.read_csv(file, header=None).values

    @staticmethod
    def write(array, file):
        pickle.dump(array, open(file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_array(file):
        file = open(file, 'rb')
        return pickle.load(file)

    def map_to_ring(self, x):
        """Maps the real elements to a ring or field"""
        a = 2**self.precision
        a = a * x
        a = np.rint(a)
        a = a.astype(int)
        a = a % self.mod
        return a

    def secret_share_onfile(self, x, file0, file1, real=True, modulo=False):
        """Secret share the secrets"""
        if not modulo:
            modulo = self.mod
        if real:
            x = self.map_to_ring(x)
        shares0 = np.random.randint(0, modulo, x.shape)
        shares1 = (x - shares0).astype(int) % modulo
        self.write(shares0, file0)
        self.write(shares1, file1)
