import numpy as np
import scipy, os
from eelbrain import *

def xs_to_Xs(xs, K):
    '''
    xs : npreds[L[N]]
    returns Xs : npreds[L[NxK]]
    '''
    Xs = []
    npreds = len(xs)
    L = len(xs[0])
    for x in xs:
        Xl = []
        for l in range(L):
            Xl.append(x_to_X(x[l], K))
        Xs.append(Xl)
    return Xs


def x_to_X(x, K, edges=0):
    '''
    x : N
    returns X : NxK
    '''
    N = len(x)
    X1 = np.zeros((K+2*edges, N))
    i=0
    for k in range(K+2*edges):
        k1 = k - edges
        if k1 < 0:
            X1[i, :] = np.concatenate([x[abs(k1):], np.zeros(abs(k1))])
        elif k1 == 0:
            X1[i, :] = x
        else:
            X1[i, :] = np.concatenate([np.zeros(k1), x[:-k1]])
        i += 1
    return X1.T

def permute_predictors_nd(preds, ip, nperm=4):
    ntrial = len(preds)
    N = len(preds[0])
    permlen = int(N/nperm)
    preds_perm = []
    for it in range(ntrial):
        predsp = []
        for p in range(nperm):
            pp = (p+1+ip)%nperm
            predsp.append(preds[it].x[pp*permlen:(pp+1)*permlen])
        preds_perm.append(NDVar(np.concatenate(predsp, axis=0), preds[it].time))
    return preds_perm

def permute_predictors_Xs(Xs, ip, nperm=4):
    ntrials = len(Xs)
    N, K = Xs[0].shape
    permlen = int(N/nperm)
    Xperms = []
    for it in range(ntrials):
        Xp = []
        for p in range(nperm):
            pp = (ip+1+p)%nperm
            Xp.append(Xs[it][pp*permlen:(pp+1)*permlen, :])
        Xperms.append(np.concatenate(Xp, axis=0))
    return Xperms


def crossval_indices():
    tests = []
    trains = []
    valids = []
    plotmap = np.zeros((12,4))
    for i in range(12):
        test = i//3
        valid = (i//3+1+i%3)%4
        train = [j for j in range(4) if j!=test and j!=valid]
        tests.append(test)
        valids.append(valid)
        trains.append(train)
        plotmap[i, test] = 1
        plotmap[i, valid] = -2
        plotmap[i, train[0]] = -1
        plotmap[i, train[1]] = -1
    return trains, valids, tests


def crossval_4fold(Y, X, yaxis=1, xaxis=0):
    cvfolds = 4
    N = Y.shape[yaxis]
    cN = int(N/cvfolds)
    cN2 = int(cN/cvfolds)
    Ysplits = []
    Xsplits = []
    for i in range(cvfolds):
        Ysplits.append(Y.take(range(i*cN,(i+1)*cN), axis=yaxis).copy())
        Xsplits.append(X.take(range(i*cN,(i+1)*cN), axis=xaxis).copy())
    trains, valids, tests = crossval_indices()
    Ytrain = []
    Yvalid = []
    Ytest = []
    Xtrain = []
    Xvalid = []
    Xtest = []
    for i in range(len(trains)):
        Ytrain.append(np.concatenate([Ysplits[trains[i][0]], Ysplits[trains[i][1]]], axis=yaxis))
        Yvalid.append(Ysplits[valids[i]])
        Ytest.append(Ysplits[tests[i]])
        Xtrain.append(np.concatenate([Xsplits[trains[i][0]], Xsplits[trains[i][1]]], axis=xaxis))
        Xvalid.append(Xsplits[valids[i]])
        Xtest.append(Xsplits[tests[i]])
    return Ytrain, Yvalid, Ytest, Xtrain, Xvalid, Xtest



def print_fit_results(logfile, tstr, subject, corrs, el, alpha_all, best_alpha):
    print(f'{subject} {tstr}')
    for k in corrs.keys():
        print(f'{k}: {np.mean(corrs[k]):.5f}, ', end='')
    print('')
    if alpha_all is not None:
        print(alpha_all)
        print(best_alpha)
    print(f'elapsed {np.floor(el/60)}:{np.floor(el%60)}')
    print('')
    with open(logfile, 'a+') as f:
        f.write(f'{subject} {tstr}\n')
        for k in corrs.keys():
            f.write(f'{k}: {np.mean(corrs[k]):.5f}, ')
        f.write('\n')
        if alpha_all is not None:
            for a in alpha_all:
                f.write(f'{a}, ')
            f.write('\n')
            f.write(f'best_alpha = {best_alpha}\n')
        f.write(f'elapsed {np.floor(el/60)}:{np.floor(el%60)}\n\n')
