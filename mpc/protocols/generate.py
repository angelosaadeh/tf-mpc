import numpy as np
import math
import pickle


def convert_to_binary(x, width):
    numbers = np.copy(x)
    length = np.prod(numbers.shape)
    original_shape = tuple(list(numbers.shape) + [width])
    numbers.shape = (1, length)
    numbers = numbers[0]
    array = [[int(b) for b in f'{num:0{width}b}'[::-1]] for num in numbers]
    array = np.array(array)
    array.shape = original_shape
    return array


def secret_share(x, mod, num_players):
    """Secret share generated random elements"""
    shares = np.random.randint(0, mod, tuple([num_players])+x.shape)
    shares[0] = ((x - np.sum(shares, 0) + shares[0]) % mod)
    return shares


def prepare(a):
    """Final packaging before sending the shares of random elements to alice and bob"""
    shares = []
    for i in range(len(a[0])):
        shares.append([a[j][i] for j in range(len(a))])
    return shares


def generate_triple(players, mod, shape):
    """Beaver's triple for a multiplication request"""
    a = np.random.randint(0, mod, shape)
    b = np.random.randint(0, mod, shape)
    c = (a*b) % mod
    a = secret_share(a, mod, players)
    b = secret_share(b, mod, players)
    c = secret_share(c, mod, players)
    shares = prepare([a, b, c])
    return shares


def generate_eda(players, size, mod, shape):
    if size == 1:
        k = 1
    else:
        k = math.floor(math.log(mod, 2)) + 2
    r = np.random.randint(0, size, shape)
    y = convert_to_binary(r, k)
    r = secret_share(r, mod, players)
    y = secret_share(y, 2, players)
    shares = prepare([r, y])
    return shares


def generate(name='all'):
    player_numb = 2
    mod = 2**60
    triple_quantity = 2**20
    triple_bin_quantity = 2**20
    eda_quantity = 2**20
    eda_bin_quantity = 2**19
    
    if name=='all':
        triple = generate_triple(2, mod, triple_quantity)
        triple_bin = generate_triple(2, 2, triple_bin_quantity)
        eda = generate_eda(2, mod, mod, eda_quantity)
        eda_bin = generate_eda(2, 1, mod, eda_bin_quantity)
        pickle.dump([triple[0], triple_bin[0], eda[0], eda_bin[0]], open('alice_rand.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump([triple[1], triple_bin[1], eda[1], eda_bin[1]], open('bob_rand.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    if name=='triple':
        triple_quantity = 2**20
        triple = generate_triple(2, mod, triple_quantity)
        pickle.dump([triple[0]], open('alice_spec_rand.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump([triple[1]], open('bob_spec_rand.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
    if name=='triple_bin':
        triple_bin_quantity = 2**20
        triple_bin = generate_triple(2, 2, triple_bin_quantity)        
        pickle.dump([triple_bin[0]], open('alice_spec_rand.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump([triple_bin[1]], open('bob_spec_rand.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    generate()
