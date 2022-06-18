import dataprep
import sklearn.linear_model
import importlib
import dataprep
import numpy as np
import scipy, multiprocessing, functools, os

def ridge_wrapper(preds, y, K, alpha_all=None, COV=None, edges=10, par=False):
    '''
    wrapper for computing ridge TRFs for multiple predictors with crossvalidation 
    inputs:
    - preds : list of 1D predictor NDVars
    - y : measurement NDVar (single- or multi-channel)
    - K : number of TRF lags
    - alpha_all : list of regularization parameters
    - COV : covariance for ridge regularization
    - edges : number of samples to remove at edges to avoid edge artifacts
    outputs:
    - avgtrf: computed TRF. The average TRF over all training splits for the best regularization parameter
    - corr: correlation on test data
    - res_dict: dictionary with all results and TRF computations
    '''
    
    importlib.reload(dataprep)
    n_pred = len(preds)
    
    # normalize predictors and form predictor matrix
    Xs = []
    Xcs = []
    xstds = []
    for ip in range(n_pred):
        x = preds[ip].x
        xstds.append(np.std(x))
        x = (x - np.mean(x))/np.std(x)
        N = len(x)
        # with additional samples at edges to avoid artifacts for fitting TRFs
        X = dataprep.x_to_X(x, K, edges=edges)
        # no additional samples for performance evaluation
        Xc = dataprep.x_to_X(x, K, edges=0)
        Xs.append(X)
        Xcs.append(Xc)
    X = np.concatenate(Xs, axis=1)
    Xc = np.concatenate(Xcs, axis=1)
    
    if len(y.x.shape) == 1: # 1-D to M-D
        y = np.asarray([y])
    else:
        y = y.x
        
    # normalize measurements
    ymeans = []
    ystds = []
    for i in range(len(y)):
        ymeans.append(np.mean(y[i]))
        ystds.append(np.std(y[i]))
        y[i] = (y[i] - np.mean(y[i]))/np.std(y[i])
        
    # prepare crossvalidation splits
    # with edge samples
    Ytrain, Yvalid, Ytest, Xtrain, Xvalid, Xtest = dataprep.crossval_4fold(y, X)
    # without edge samples
    Ytrain, Yvalid, Ytest, Xctrain, Xcvalid, Xctest = dataprep.crossval_4fold(y, Xc)
    
    M = len(Ytrain[0]) # number of channels
    
    if alpha_all is None:
        alpha_all = [10**5]
    print(alpha_all)

    ridge_cv_par = functools.partial(ridge_cv, Ytrain.copy(), Yvalid.copy(), Xtrain.copy(), Xcvalid.copy(), COV, n_pred, K, edges)
    
    trfscv_all = []
    corrscv_all = []
    par = False
    if par: # parallel processing over regularization parameters
        pool = multiprocessing.Pool(os.cpu_count())
        presult = list(pool.map(ridge_cv_par, alpha_all))
        for p in presult:
            trfscv_all.append(p[0])
            corrscv_all.append(p[1])
    else:
        for ii, alpha in enumerate(alpha_all):
            trf, corr = ridge_cv_par(alpha)
            trfscv_all.append(trf)
            corrscv_all.append(corr)
    
    # extract average TRFs and metrics over all regularization parameters
    avgtrfs_all = []
    avgcorrs_all = []
    maxtrfs_all = []
    maxcorrs_all = []
    for trfscv, corrscv in zip(trfscv_all, corrscv_all):
        avgtrfs_all.append(np.mean(np.asarray(trfscv), axis=0))
        avgcorrs_all.append(np.mean(np.asarray(corrscv), axis=0))
        maxidx = np.argmax([np.mean(c) for c in corrscv])
        maxtrfs_all.append(trfscv[maxidx])
        maxcorrs_all.append(corrscv[maxidx])
        
    # best regularization parameter
    bidx = np.argmax([np.mean(a) for a in avgcorrs_all])
    trfscv = trfscv_all[bidx]
    
    # model fits
    # compute correlation on test data
    # using average TRF over all training splits
    avgtrf = np.mean(np.asarray(trfscv), axis=0)
    ypreds = []
    yactuals = []
    for i in range(4):
        atrf = np.mean(np.asarray(trfscv[i*3:(i+1)*3]), axis=0)
        x1 = Xctest[i*3]
        yactuals.append(Ytest[i*3])
        yp1 = []
        for im in range(M):
            yp1.append(x1@atrf[im])
        ypreds.append(np.asarray(yp1))
    ypred = np.concatenate(ypreds, axis=1)
    yactuals = np.concatenate(yactuals, axis=1)
    corr = []
    for im in range(M):
        corr.append(np.corrcoef(yactuals[im], ypred[im])[0,1])

    print(f'ridge best {bidx+1}/{len(alpha_all)} {alpha_all[bidx]} {np.mean(corr):.5f}')
    
    # make dictionary with all results
    res_dict = dict(trfscv_all=trfscv_all, ypred=ypred, bidx=bidx, avgtrfs_all=avgtrfs_all, 
                    avgcorrs_all=avgcorrs_all, maxtrfs_all=maxtrfs_all, maxcorrs_all=maxcorrs_all, 
                    best_alpha=alpha_all[bidx], alpha_all=alpha_all, corr=corr, ystds=ystds, xstds=xstds)
    
    return avgtrf, corr, res_dict


def ridge_cv(Ytrain, Yvalid, Xtrain, Xvalid, COV, n_pred, K, edges, alpha):
    '''
    fit ridge using crossvalidation
    inputs:
    - Ytrain: list of training crossvalidation measurement splits
    - Yvalid: list of validation crossvalidation measurement splits
    - Xtrain: list of training crossvalidation predictor matrix splits
    - Xvalid: list of validation crossvalidation predictor matrix splits
    - COV: covariance for ridge regularization
    - n_pred: number of predictors
    - K: number of TRF lags
    - edges: samples to remove due to edge artifacts
    - alpha: regularization parameter
    outputs:
    - trfscv : list of trfs for each split
    - corrscv : list of model fit correlations for each split
    '''
    
    corrscv = [] # model fits
    trfscv = [] # trfs
    M = len(Ytrain[0]) # channels
    for cv in range(len(Ytrain)):
        ycv = Ytrain[cv]
        Xcv = Xtrain[cv]
        yvalcv = Yvalid[cv]
        Xvalcv = Xvalid[cv]

        ridge_1Dm = functools.partial(ridge_1D, ycv, Xcv, alpha, K, COV, edges, n_pred, yvalcv, Xvalcv)
        
        # fit independently at each channel
        trfs = []
        corrs = []
        for m in range(M):
            p = ridge_1Dm(m)
            trfs.append(p[0])
            corrs.append(p[1])

        corrscv.append(corrs)
        trfscv.append(trfs)
    return trfscv, corrscv


def ridge_1D(y, X, alpha, K, COV, edges, n_pred, yval, Xval, m):
    '''
    compute ridge solution
    inputs:
    - y : measurements
    - X : predictor matrix
    - alpha : regularization parameter
    - K : number of trf lags
    - COV : noise covariance
    - edges : edge samples used to fit TRF and avoid edge artifacts
    - n_pred : number of predictors (used to remove edges)
    - yval : validation measurements
    - Xval : validation predictor matrix
    - m : channel number
    outputs:
    - trf : TRF
    - corr : model fit correlation
    '''
    y1 = y[m].copy()
    alpha = alpha * np.trace(X.T@X) / X.shape[0] # scalebased on predictor matrix
    if COV is None:
        COV = np.eye(X.shape[1])
    trf1 = scipy.linalg.pinv(X.T@X+alpha*COV)@X.T@y1
    
    # remove edges in TRF
    trf11 = []
    for ip in range(n_pred):
        trf11.append(trf1[ip*(K+2*edges)+edges:(ip+1)*(K+2*edges)-edges])
    trf = np.concatenate(trf11, axis=0)
    
    # model fit on validation data
    corr = np.corrcoef(yval[m], Xval@trf)[0,1]
    
    return trf, corr
