import numpy as np
from matplotlib import pyplot as plt
import time
import sys
sys.path.append("..")
from mpc.data import data


def normalize(vector):
    #return 0.5*vector+0.25
    temp = np.copy(vector)
    a = max(np.max(vector), 1)
    i = min(np.min(vector), -1)
    temp = (vector - i) / (a - i)
    return 2*temp-1


def cosine(votes, iterations, target):
    v = votes.transpose()
    t = (v == 1)
    t_ = (v == -1)
    nv, nq = v.shape
    error = []

    N = np.sum(v != 0, 1)  # Number of {votes!=0}
    trust = (np.sum(t, 1) - np.sum(t_, 1)) / N
    trust.shape = (nv, 1)
    N.shape = trust.shape
    y = np.ones((nq, 1))
    target.shape = y.shape
    error.append(np.sum(np.abs(target - (y >= 0))))
    eta = 0.2
    p = []
    q = []
    for iteration in range(iterations):
        wall = time.time()
        cpu = time.process_time()
        pos = np.matmul(t, y)
        neg = np.matmul(t_, y)
        norm = np.matmul((v != 0), y ** 2) * N
        norm = np.sqrt(norm)
        new_trust = (pos - neg) / norm
        trust = trust * (1 - eta) + new_trust * eta
        if iteration!=0:
            trust = normalize(trust)

        pos = np.matmul(t.transpose(), (trust**3))
        neg = np.matmul(t_.transpose(), (trust**3))
        norm = pos + neg
        y = (pos - neg) / norm

        error.append(np.sum(np.abs(target - (y >= 0))))
        p.append(time.time() - wall)
        q.append(time.process_time() - cpu)
        
    print('The average wall and cpu time are: ', np.mean(p), np.mean(q))
    return y, trust, np.array(error)


def cosine_mpc(mpc, votes, iterations, target):
    v = votes.transpose()
    v2 = mpc.mul(v,v)

    t = mpc.mul_const(mpc.add(v2,v), 0.5)
    t_ = mpc.mul_const(mpc.subs(v2,v), 0.5)

    nv, nq = v.shape
    error = []

    N = mpc.sum(v2, 1) # Number of {votes!=0}
    srN = mpc.new_1_sqrt(N)
    trust = mpc.mul(mpc.subs(mpc.sum(t, 1), mpc.sum(t_, 1)), mpc.new_inverse(N))
    trust.shape = (nv, 1)
    N.shape = trust.shape

    if mpc.identity == 'alice':
        y = np.ones((nq,1))
        y = mpc.secret_share(y)
        time.sleep(1)
        amin = np.array([-1])
        amin = mpc.secret_share(amin)
        time.sleep(1)
        amax = np.array([1])
        amax = mpc.secret_share(amax)
        time.sleep(1)
    else:
        y = mpc.receive_shares()
        amin = mpc.receive_shares()
        amax = mpc.receive_shares()
    target.shape = y.shape

    e = np.sum(np.abs(target - (mpc.reconstruct(y) >= 0)))
    print(e)
    error.append(e)
    eta = 0.2
    p = []
    q = []
    for iteration in range(iterations):
    
        #print('-----', iteration, '------')
        tempp = time.time()
        tempq = time.process_time()
        pos = mpc.matmul(t,y)
        neg = mpc.matmul(t_,y)
        pre_norm = mpc.new_1_sqrt(mpc.matmul(v2, mpc.mul(y, y)))
        srN.shape = pre_norm.shape
        norm = mpc.mul(pre_norm, srN)
        new_trust = mpc.mul(mpc.subs(pos, neg), norm)
        trust = mpc.add(mpc.mul_const(trust, 1-eta) , mpc.mul_const(new_trust, eta))

        if iteration!=0:
            trust = mpc.normalize(trust, amin, amax, free=False, unitary=False)

        t3 = trust
        num = mpc.matmul(v.transpose(), t3)
        norm = mpc.matmul(v2.transpose(), t3)
        
        y = mpc.mul(mpc.mul(num, norm), mpc.new_inverse(mpc.square(norm)))

        e = np.sum(np.abs(target - (mpc.reconstruct(y) >= 0)))
        if iteration != 0:
            p.append(time.time()-tempp)
            q.append(time.process_time()-tempq)
        error.append(e)

    print('The average wall and cpu time are: ', np.mean(p), np.mean(q))
    return mpc.reconstruct(y), mpc.reconstruct(trust), np.array(error)


def function(mpc, v):
    v = v[:120,:]
    
    nv, nq = v.transpose().shape
    np.set_printoptions(suppress=True)
    v_clear = mpc.reconstruct(v)
    target = data.Unpack().get_array('data/mnist.target')[:120,:]
    print('Running algorithm in plain')
    n = cosine(v_clear, 5, target)
    print('Running algorithm with MPC')
    m = cosine_mpc(mpc, v, 5, target)

    print('Similar answers', np.sum(((m[0]) * (n[0]) >= 0)), '/', nq)
    print('Number of errors over the iterations in MPC:', m[-1])
    print('Number of errors over the iterations in plain:', n[-1])
    print('Error graphs will be saved in the directory')

    if mpc.identity == 'alice':
        plt.figure(figsize=(45, 10))
        plt.subplot(1, 2, 1)
        plt.gca().set_title('Prediction error of the truth value for each query', fontsize=30)
        plt.plot([i for i in range(1, nq+1)], np.abs(m[0]-n[0]), 'kv', linewidth=0.5)
        plt.plot([np.mean(np.abs(m[0]-n[0]))] * (nq+1), 'b--', linewidth=0.5, label='Mean prediction error')
        plt.plot([np.median(np.abs(m[0]-n[0]))] * (nq+1), 'r--', linewidth=0.5, label='Median prediction error')
        plt.ylabel("Truth prediction error", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(1, 2, 2)
        plt.gca().set_title('Prediction error of the trust value for each voter', fontsize=30)
        plt.plot([i for i in range(1, 16)], np.abs(m[1] - n[1]), 'kv', linewidth=0.5)
        plt.plot([np.mean(np.abs(m[1] - n[1]))] * 16, 'b--', linewidth=0.5, label='Mean prediction error')
        plt.plot([np.median(np.abs(m[1] - n[1]))] * 16, 'r--', linewidth=0.5, label='Median prediction error')
        plt.ylabel("Trust prediction error", fontsize=30)
        plt.xlabel("Voter ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.savefig('errors.eps', format='eps')

    '''
    if mpc.identity == 'alice':
        y = np.array([100,1000,10**4,10**5,10**6])
        y = mpc.secret_share(y)
        time.sleep(1)
    else:
        y = mpc.receive_shares()

    z = mpc.sqrt(y)
    z = mpc.reconstruct(z)
    print(z)

    z = mpc.log_high(y)
    z = mpc.reconstruct(z)
    print(z)
    '''
    return 0
