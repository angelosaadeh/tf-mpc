import numpy as np
from matplotlib import pyplot as plt
import time
import sys
sys.path.append("..")
from mpc.data import data
from time import process_time as pt


def norm(vector):
    temp = np.copy(vector)
    a = max(np.max(vector), 1)
    i = min(np.min(vector), 0)
    temp = (vector - i) / (a - i)
    return temp


def est3_clear(v, iterations, target):
    nq, nv = v.shape
    trust = 0.4 * np.ones((1, nv))
    diff = 0.1 * np.ones((nq, 1))
    target.shape = (nq, 1)
    y = np.ones((nq, 1))

    print('Initial plaintext accuracy: ',np.mean((y >= 0.5) != target))
    t = (v == 1)
    t_ = (v == -1)
    p = []
    q = []
    
    for it in range(iterations):
        tempp = time.time()
        tempq = time.process_time()
        n = np.sum(v ** 2, 1)
        pos = np.sum(t * (1 - np.matmul(diff, trust)), 1)
        neg = np.sum(t_ * np.matmul(diff, trust), 1)
        y = (pos + neg) / n
        y.shape = (nq, 1)
        y = norm(y)

        n = np.sum(v ** 2, 1)
        pos = np.sum(t * np.matmul((1 - y), 1 / trust), 1)
        neg = np.sum(t_ * np.matmul(y, 1 / trust), 1)
        n.shape = pos.shape
        diff = (pos + neg) / n
        diff.shape = (nq, 1)
        diff = norm(diff)

        n = np.sum(v ** 2, 0)
        pos = np.matmul(t.transpose(), (1 - y) / diff)
        neg = np.matmul(t_.transpose(), y / diff)
        n.shape = pos.shape
        trust = (pos + neg) / n
        trust.shape = (1, nv)
        trust = norm(trust)
        p.append(time.time() - tempp)
        q.append(time.process_time() - tempq)

        print('accuracy mpc', np.mean((y >= 0.5) != target))
        
    print('Average Wall and CPU time:', np.mean(p), np.mean(q))
    return y, trust, diff


def est3_mpc(mpc, v, iterations, target):
    nq, nv = v.shape

    if mpc.identity == 'alice':
        trust = 0.4*np.ones((1,nv))
        trust = mpc.secret_share(trust)
        time.sleep(1)
        amin = np.array([0])
        amin = mpc.secret_share(amin)
        time.sleep(1)
        amax = np.array([1])
        amax = mpc.secret_share(amax)
        time.sleep(1)
        diff = 0.1*np.ones((nq,1))
        diff = mpc.secret_share(diff)
    else:
        trust = mpc.receive_shares()
        amin = mpc.receive_shares()
        amax = mpc.receive_shares()
        diff = mpc.receive_shares()


    y = np.ones((nq,1))
    target.shape = y.shape

    print('Initial MPC accuracy: ', np.mean((y >= 0.5) != target))
    ### Compute the truth matrix where v==1 knowing that there are no zeros in v
    v_2 = mpc.mul_const(v, 1/2)
    k = mpc.mul(v, v)
    v2 = mpc.mul_const(k, 1 / 2)
    t =  mpc.add(v2, v_2) #=(v==1)
    t_ = mpc.subs(v2, v_2) #(v==-1)
    ###
    n = mpc.new_inverse(mpc.sum(mpc.mul_const(v2, 2), 1))
    m = mpc.new_inverse(mpc.sum(mpc.mul_const(v2, 2), 0))
    p = []
    q = []
    for it in range(iterations):
        tempp = time.time()
        tempq = time.process_time()
        pos = mpc.sum(mpc.mul(t, mpc.add_const(-mpc.matmul(diff, trust), 1)), 1)
        neg = mpc.sum(mpc.mul(t_, mpc.matmul(diff, trust)), 1)
        y = mpc.mul(pos + neg, n)
        y.shape = (nq, 1)
        y = mpc.normalize(y, amin, amax, free=True, unitary=True)

        temp = mpc.new_inverse(trust)
        pos = np.sum(mpc.mul(t, mpc.matmul(mpc.add_const(-y, 1), temp)), 1)
        neg = mpc.sum(mpc.mul(t_, mpc.matmul(y, temp)), 1)
        diff = mpc.mul(pos + neg, n)
        diff.shape = (nq, 1)
        diff = mpc.normalize(diff, amin, amax, free=True, unitary=True)

        temp = mpc.new_inverse(diff)
        pos = mpc.matmul(t.transpose(), mpc.mul(mpc.add_const(-y, 1), temp))
        neg = mpc.matmul(t_.transpose(), mpc.mul(y, temp))
        m.shape = pos.shape
        trust = mpc.mul(pos + neg, m)
        trust.shape = (1, nv)
        trust = mpc.normalize(trust.transpose(), amin, amax, free=True, unitary=True).transpose()
        
        p.append(time.time()-tempp)
        q.append(time.process_time()-tempq)
        print('accuracy mpc ', np.mean(np.abs((mpc.reconstruct(y) > 0.5) - target)))
    print('Average Wall and CPU time:', np.mean(p), np.mean(q))
    return mpc.reconstruct(y), mpc.reconstruct(trust), mpc.reconstruct(diff)

def function(mpc, v):

    np.set_printoptions(suppress=True)
    v_clear = mpc.reconstruct(v)
    target = data.Unpack().get_array('data/hubdub.target')

    a = est3_mpc(mpc, v.transpose(), 6, target)
    print('-------clear--------')
    b = est3_clear(v_clear.transpose(), 6, target)

    print('error', np.sum(((a[0] - 0.5) * (b[0] - 0.5)) > 0))

    ay = a[0]-b[0]
    at = a[1]-b[1]
    ad = a[2]-b[2]

    ay.shape = (830,)
    at.shape = (471,)
    ad.shape = (830,)

    if mpc.identity == 'alice':
        plt.figure(figsize=(45, 10))
        plt.subplot(1, 3, 1)
        plt.gca().set_title('Prediction error of the truth value for each query', fontsize=30)
        plt.plot([i for i in range(1, 831)], np.abs(ay), 'kv')
        plt.plot([np.mean(np.abs(ay))] * 831, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(ay))] * 831, 'r--', label='Median prediction error')
        plt.ylabel("Truth prediction error", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(1, 3, 2)
        plt.gca().set_title('Prediction error of the difficulty factor for each query', fontsize=30)
        plt.plot([i for i in range(1, 831)], np.abs(ad), 'kv')
        plt.plot([np.mean(np.abs(ad))] * 831, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(ad))] * 831, 'r--', label='Median prediction error')
        plt.ylabel("Difficulty prediction error", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(1, 3, 3)
        plt.gca().set_title('Prediction error of the trust value for each voter', fontsize=30)
        plt.plot([i for i in range(1, 472)], np.abs(at), 'kv')
        plt.plot([np.mean(np.abs(at))] * 472, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(at))] * 472, 'r--', label='Median prediction error')
        plt.ylabel("Trust prediction error", fontsize=30)
        plt.xlabel("Voter ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.savefig('tests-error.eps', format='eps')

    return 0
