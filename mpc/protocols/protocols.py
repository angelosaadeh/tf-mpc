"""This class contains the MPC functions that are common to all users"""
import pickle
import numpy as np
import math
import mpc.protocols.communications as communications
import mpc.protocols.generate as generate
import time


class Protocols(communications.Communication):
    """Protocols for MPC"""
    def __init__(self, identity, partner, mod, precision, length):
        """Identity is a string equals to alice or bob"""
        super().__init__(length)
        self.identity = identity
        self.partner = partner
        self.mod = mod
        self.precision = precision
        self.triple, self.triple_bin, self.eda, self.eda_bin = self.get_crypto()
        self.interactions = 0
        self.number_mul_bin = 0
        self.number_mul = 0
        self.qty_mul_bin = 0
        self.qty_mul = 0

    def get_crypto(self):
        if self.identity == 'alice':
            file = open('../mpc/protocols/alice_rand.p', 'rb')
        else:
            file = open('../mpc/protocols/bob_rand.p', 'rb')
        triple, triple_bin, eda, eda_bin = pickle.load(file)
        file.close()
        return triple, np.array(triple_bin), eda, eda_bin


    def get_specific_crypto(self):
        if self.identity == 'alice':
            file = open('../mpc/protocols/alice_spec_rand.p', 'rb')
        else:
            file = open('../mpc/protocols/bob_spec_rand.p', 'rb')
        crypto = pickle.load(file)[0]
        file.close()
        return crypto

    def map_to_ring(self, x):
        """Maps the real elements to a ring or field"""
        a = 2**self.precision
        a = a * x
        a = np.rint(a)
        a = a.astype(int)
        a = a % self.mod
        return a

    def secret_share(self, x, real=True, modulo=False):
        """Secret share the secrets"""
        if not modulo:
            modulo = self.mod
        if real:
            x = self.map_to_ring(x)
        shares_partner = np.random.randint(0, modulo, x.shape)
        shares_identity = (x - shares_partner).astype(int) % modulo
        self.send(self.partner, shares_partner)
        return shares_identity

    def secret_share_bin(self, x):
        """Secret share the secrets"""
        shares_partner = np.random.randint(0, 2, x.shape)
        shares_identity = (x - shares_partner).astype(int) % 2
        self.send(self.partner, shares_partner)
        return shares_identity

    def receive_shares(self):
        """To be used after every secret_share on the receiver's side"""
        return self.get(self.partner)

    def reconstruct(self, x):
        """Reconstruct a secret"""
        if self.identity == 'alice':
            self.send(self.partner, x)
            y = self.get(self.partner)
        if self.identity == 'bob':
            y = self.get(self.partner)
            self.send(self.partner, x)
        z = (x+y) % self.mod
        return (z > (self.mod/2)) * (-self.mod / 2**self.precision) + (z/2**self.precision)

    def reveal(self, x):
        """Open an element without converting it to a real number, ie keep it as a ring element"""
        if self.identity == 'alice':
            self.send(self.partner, x)
            y = self.get(self.partner)
        if self.identity == 'bob':
            y = self.get(self.partner)
            self.send(self.partner, x)
        return (x+y) % self.mod

    def reveal_bin(self, x):
        """Open an element without converting it to a real number, ie keep it as a ring element"""
        if self.identity == 'alice':
            self.send(self.partner, x)
            y = self.get(self.partner)
        if self.identity == 'bob':
            y = self.get(self.partner)
            self.send(self.partner, x)
        return (x+y) % 2

    def add(self, x, y):
        """Addition of two secrets"""
        return (x+y) % self.mod

    @staticmethod
    def add_bin(x, y):
        return (x+y) % 2

    def subs(self, x, y):
        """Subtraction of two secrets"""
        return (x-y) % self.mod

    def add_const(self, x, k, real=True):
        """Addition with a constant that should be mapped to the field if it is real"""
        if self.identity == 'alice':
            if real:
                k = self.map_to_ring(k)
            x = (x+k) % self.mod
        return x % self.mod

    def add_const_bin(self, x, k):
        if self.identity == 'alice':
            x = (x+k) % 2
        return x % 2

    def new_truncate_bit(self, x):
        """Truncate one bit"""
        if self.identity == 'alice':
            return x >> self.precision
        return self.mod - ((self.mod-x) >> self.precision)

    def truncate(self, x, bits, old):
        """Truncate the required number of bits"""
        if old:
            for i in range(bits):
                print("Old truncate is no longer available")
        else:
            x = self.new_truncate_bit(x)
        return x

    def get_triple_bin(self, x):
        quantity = np.prod(x.shape)
        self.qty_mul_bin += quantity
        if quantity > len(self.triple_bin[0]):
            print("-")
            if self.identity == 'alice':
                generate.generate(name='triple_bin')
                self.send(self.partner, 'go')
                self.triple_bin = np.array(self.get_specific_crypto())
            if self.identity == 'bob':
                _ = self.get(self.partner)
                self.triple_bin = np.array(self.get_specific_crypto())
        a = self.triple_bin[0][0:quantity]
        b = self.triple_bin[1][0:quantity]
        c = self.triple_bin[2][0:quantity]
        a.shape = x.shape
        b.shape = x.shape
        c.shape = x.shape
        #self.triple_bin = [vector[quantity:] for vector in self.triple_bin]
        return a, b, c

    def get_triple(self, x):
        quantity = np.prod(x.shape)
        self.qty_mul += quantity
        if quantity > len(self.triple[0]):
            print("--")
            if self.identity == 'alice':
                generate.generate(name='triple')
                self.send(self.partner, 'go')
                self.triple = self.get_specific_crypto()
            if self.identity == 'bob':
                _ = self.get(self.partner)
                self.triple = self.get_specific_crypto()
        a = self.triple[0][0:quantity]
        b = self.triple[1][0:quantity]
        c = self.triple[2][0:quantity]
        a.shape = x.shape
        b.shape = x.shape
        c.shape = x.shape
        #self.triple = [vector[quantity:] for vector in self.triple]
        return a, b, c

    def get_eda(self, x):
        quantity = np.prod(x.shape)
        r = self.eda[0][0:quantity]
        r_bin = self.eda[1][0:quantity]
        r_bin = self.eda[1][0:quantity]
        r.shape = x.shape
        r_bin.shape = x.shape + tuple([math.floor(math.log(self.mod, 2)) + 2])
        #self.eda = [vector[quantity:] for vector in self.eda]
        return r, r_bin

    def get_eda_bin(self, x):
        quantity = np.prod(x.shape)
        r = self.eda_bin[0][0:quantity]
        r_bin = self.eda_bin[1][0:quantity]
        r.shape = x.shape
        r_bin.shape = x.shape
        #self.eda_bin = [vector[quantity:] for vector in self.eda_bin]
        return r, r_bin

    def mul_bin(self, x, y):
        """Element-wise multiplication of secrets"""
        self.number_mul_bin += 1
        a, b, c = self.get_triple_bin(x)
        d = self.add_bin(x, a)
        e = self.add_bin(y, b)
        d, e = self.reveal_bin(np.array([d, e]))
        z = self.add_const_bin(c + d * b + e * a, e * d)
        return z

    def mul(self, x, y, real=True, old=False):
        """Element-wise multiplication of secrets"""
        self.number_mul += 1
        a, b, c = self.get_triple(x)
        d = self.subs(x, a)
        e = self.subs(y, b)
        d, e = self.reveal(np.array([d, e]))
        z = self.add_const(c + d * b + e * a, e * d, real=False)
        if real:
            z = self.truncate(z, self.precision, old)
        return z

    def matmul(self, x, y):
        """Multiplication of two matrices"""
        if len(x.shape) != len(y.shape) or len(x.shape) != 2:
            print('ERROR: They should be 2x2 matrices')
            return 0
        x_i, x_j = x.shape
        y_i, y_j = y.shape
        if x_j != y_i:
            print('ERROR: Shapes do not match')
            return 0
        x_repeated = np.tile(x, (1, y_j))
        x_repeated.shape = (x_i, y_j, x_j)
        y_repeated = np.tile(y.transpose(), (x_i, 1))
        y_repeated.shape = (x_i, y_j, x_j)
        z_repeated = self.mul(x_repeated, y_repeated)
        z = np.sum(z_repeated, 2) % self.mod
        return z

    @staticmethod
    def convert_to_bin(x, width):
        numbers = np.copy(x)
        length = np.prod(numbers.shape)
        original_shape = tuple(list(numbers.shape)+[width])
        numbers.shape = (1, length)
        numbers = numbers[0]
        array = [[int(b) for b in f'{int(num):0{width}b}'[::-1]] for num in numbers]
        array = np.array(array)
        array.shape = original_shape
        return array

    def convert_from_bin(self, x):
        r, r_bin = self.get_eda_bin(x)
        r_bin.shape = x.shape
        y = self.reveal_bin(self.add_bin(x, r_bin))
        z = self.add_const(r - 2 * y * r, y, real=False)
        return z

    def or_bin(self, x, y):
        z = self.add_bin(x, y)
        m = self.mul_bin(x, y)
        return self.add_bin(z, m)

    def prefix_or(self, x):
        l = math.log2(len(x[0]))
        while l != int(l):
            x = np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)
            l = math.log2(len(x[0]))
        k = np.prod(x.shape)
        l = int(len(x[0]) / 2)
        if l == 1:
            a = x[:, :l]
            b = x[:, l:]
            c = self.or_bin(a, b)
            r = np.concatenate((c, b), axis=1)
            return r
        else:
            x.shape = (int(k / l), l)
            r = self.prefix_or(x)
            a = np.array([r[2 * i] for i in range(int(k / l / 2))])
            b = np.array([r[2 * i + 1] for i in range(int(k / l / 2))])
            b_0 = np.tile(np.array([b[:, 0]]).transpose(), (1, l))
            c = self.or_bin(b_0, a)
            return np.concatenate((c, b), axis=1)

    def less_than_old(self, c, x):
        c_copy = np.copy(c)
        y = self.add_const_bin(x, c_copy)
        shape = c_copy.shape
        final_shape = tuple(shape[:-1])
        y.shape = (np.prod(final_shape), shape[-1])
        y = y.transpose()
        z = y[-1]
        z.shape = (1, len(y[0]))
        w = z[0]
        w.shape = (1, len(y[0]))
        for i in range(len(y) - 1):
            bib = self.or_bin(y[len(y) - i - 2], z[0])
            z = np.insert(z, 0, bib , axis=0)
            w = np.insert(w, 0, (z[0] + z[1]) % 2, axis=0)
        c_copy.shape = w.transpose().shape
        w = np.sum(w.transpose() * (1-c_copy), 1) % 2
        w.shape = final_shape
        return w

    def mul_const(self, x, c, real=True):
        if real:
            c = self.map_to_ring(c)
        y = (c*x) % self.mod
        if real:
            y = self.truncate(y, self.precision, old=False)
        return y

    def rabbit_compare(self, x, c, real=True, old=True): # returns x-c>=0
        """ Add description here """
        if real:
            x = self.add_const(x, -c)
        else:
            x = self.add_const(x, -c, real=False)
        c = self.mod/2
        r, r_bin = self.get_eda(x)
        a = self.add(x, r)
        a = self.reveal(a)
        b = self.subs(a, c)
        k = math.floor(math.log(self.mod, 2)) + 2
        b_bin = self.convert_to_bin(b, k)
        a_bin = self.convert_to_bin(a, k)
        # w1 = self.less_than(a_bin, r_bin) #on veut a<r
        # w2 = self.less_than(b_bin, r_bin) #on veut b<r
        if old==True:
            w = self.less_than_old(np.concatenate((a_bin, b_bin)), np.concatenate((r_bin, r_bin)))  # on veut a<r et b<r
        else:
            print("New less than circuit not available in this version")
        w1 = w[:a_bin.shape[0]]
        w2 = w[a_bin.shape[0]:]
        w3 = b < (self.mod-c)
        w_bin = self.add_const_bin(w1-w2, w3)
        w_bin = self.add_const_bin(-w_bin, 1)
        w = self.convert_from_bin(w_bin)
        w = self.map_to_ring(w)
        return w

    def exp(self, x, iterations=8):
        a = self.add_const(self.mul_const(x, 2**-iterations), 1)
        for i in range(iterations):
            a = self.mul(a, a)
        return a

    def bilan(self):
        print("Interactions: ", self.interactions)
        print("Multiplications: ", self.number_mul)
        print("Multiplications binaires: ", self.number_mul_bin)
        print("Triplets de multiplications utilisés: ", self.qty_mul)
        print("Triplets de multiplications binaires utilisés: ", self.qty_mul_bin)

    def reset(self):
        self.interactions = 0
        self.number_mul_bin = 0
        self.number_mul = 0
        self.qty_mul_bin = 0
        self.qty_mul = 0

    def sum(self, x, b):
        return np.sum(x, b) % self.mod

    def sum_all(self, x):
        return np.sum(x) % self.mod

    def cube(self, x):
        y = self.mul(x, x)
        return self.mul(y, x)

    def square(self, x):
        return self.mul(x, x)

    def minmax(self, array, amin=None, amax=None):
        '''The array should be of shape (n,1)'''
        if amin==None:
            amin = array[0]
        if amax==None:
            amax = array[0]
        for i in range(len(array)):
            c = self.rabbit_compare(self.subs(amax, array[i]), 0)
            amax = self.add(self.mul(c, self.subs(amax, array[i])), array[i])
            c = self.rabbit_compare(self.subs(amin, array[i]), 0)
            amin = self.add(self.mul(c, self.subs(array[i], amin)), amin)
        return amin, amax

    def normalize(self, array, amin=None, amax=None, free=False, unitary=False):
        '''The array should be of shape (n,1)'''
        if free:
            return self.add_const(self.mul_const(array, 0.5), 0.25, real=True)
        amin, amax = self.minmax(array, amin, amax)
        denom = self.new_inverse(self.subs(amax, amin))
        temp = self.subs(array, amin)
        temp = self.mul(temp, np.tile(denom,(temp.shape[0],1)))
        if unitary:
            return temp
        return self.add_const(self.mul_const(temp, 2), -1)

    def new_inverse(self, x):
        y = self.exp(self.add_const(-x,0.5))
        y = self.add_const(self.mul_const(y,3),0.003)
        for i in range(15):
            y = self.mul(y, self.add_const(self.mul(-x,y), 2))
        return y

    def new_1_sqrt(self, x):
        y = self.exp(self.add_const(self.mul_const(x,-0.7),-0.6))
        y = self.add_const(self.mul_const(y,5),0.003)
        for i in range(15):
            y = self.mul(self.mul_const(y, 0.5), self.add_const(self.mul(-x,self.mul(y,y)), 3))
        return y
