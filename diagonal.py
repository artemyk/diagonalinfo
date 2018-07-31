from __future__ import print_function
import numpy as np
import scipy.optimize

def entropy(pX):
    assert(np.isclose(pX.sum(), 1))
    v = pX[pX > 0]
    return -v.dot(np.log2(v))

def mutual_info(pcond, pX):
    assert(np.allclose(pcond.sum(axis=1), 1))
    assert(np.isclose(pX.sum(), 1))

    hcond = 0.0
    for rndx, r in enumerate(pcond):
        hcond += pX[rndx] * entropy(r)

    mi = entropy(pX.dot(pcond)) - hcond

    if np.isclose(mi, 0):
        return 0.0
    else:
        return mi

def get_diagonal_channel(pcond, pX):
    assert(np.allclose(pcond.sum(axis=1), 1))
    assert(np.isclose(pX.sum(), 1))

    n = len(pX)

    diagconstraints = np.diag(pcond).copy()
    
    if diagconstraints.sum() <= 1:
        diagconstraints /= diagconstraints.sum()
        return pcond*0 + diagconstraints

    def qprime(a):
        return np.array([ (a+pX[i]-np.sqrt((a+pX[i])**2-4*a*pX[i]*diagconstraints[i]))/(2*a) for i in range(n) ])
    def f(a):
        return qprime(a).sum() - 1

    root = scipy.optimize.bisect(f, 1e-10, 1)
    foundvals = qprime(root)
    
    r = foundvals[None,:] * ((1. - diagconstraints)/(1. - foundvals))[:,None]
    np.fill_diagonal(r, diagconstraints)
    return r

if __name__ == "__main__":
    np.random.seed(1234)
    n = 25
    pjoint = np.random.uniform(size=(n,n))
    pjoint /= pjoint.sum()
    pX = pjoint.sum(axis=1)
    pcond = pjoint / np.atleast_2d(pX).T

    print('MI(X;Y): %0.6f bits' % mutual_info(pcond, pX))

    channel = get_diagonal_channel(pcond, pX)

    print('DI(X;Y): %0.6f bits' % mutual_info(channel, pX))
    