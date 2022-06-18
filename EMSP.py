import numpy as np
import scipy
import os
from eelbrain import *
import dataprep
import importlib, multiprocessing, os, functools


def SP_singlechannel_wrapper(preds, y, K, wins, signs, basis, n_comp=3):
    '''
    wrapper for computing TRF using Subspace Pursuit and cross-validation
    
    inputs
    - preds : list of predictor vector NDVar
    - y : single channel measurement vector NDVar
    - K : length of TRF
    - wins : windows for TRF components
    - signs : signs of TRF components
    - basis : TRF basis dictionary
    - n_comp : number of components in the TRF for each predictor
    
    outputs
    - avgtrf : average TRF across training folds
    - corr : correlation on test data
    - SPdict : dictionary with training TRFs and correlations
    '''

    importlib.reload(dataprep)
    
    # standardize predictor vectors and convert to matrix
    n_pred = len(preds)
    Xs = []
    xstds = []
    for ip in range(n_pred):
        x = preds[ip].x
        xstds.append(np.std(x))
        x = (x - np.mean(x))/np.std(x)
        N = len(x)
        X = dataprep.x_to_X(x, K, edges=0)
        Xs.append(X)
    X = np.concatenate(Xs, axis=1)
    
    # standardize measurements
    if len(y.shape) == 1:  # 1-D to M-D
        y = np.asarray([y])
    ymeans = []
    ystds = []
    for i in range(len(y)):
        ymeans.append(np.mean(y[i]))
        ystds.append(np.std(y[i]))
        y[i] = (y[i] - np.mean(y[i])) / np.std(y[i])
    
    # make data folds for crossvalidation
    Ytrain, Yvalid, Ytest, Xtrain, Xvalid, Xtest = dataprep.crossval_4fold(y, X)
    
    # initialize
    corrscv = []
    trfscv = []
    latscv = []
    ampscv = []
    J = n_comp * n_pred # 3 components in the TRF for each predictor
    
    # fit SP TRFs for each cv fold
    for cv in range(len(Ytrain)):
        YTr = Ytrain[cv]
        XTr = Xtrain[cv]
        YTe = Yvalid[cv]
        XTe = Xvalid[cv]
        # use independent SP at this single channel
        SPdict = SP_multichannel_independent(YTr, XTr, YTe, XTe, basis, wins, J, signs)
        trfscv.append(SPdict['trfs'][0])
        corrscv.append(SPdict['errs']['corr'])
        latscv.append(SPdict['lats'][0])
        ampscv.append(SPdict['amps'][0])

    # compute predicted test data using appropriate training trf
    ypreds = []
    yactuals = []
    ncv = 4
    ncv2 = 3
    for i in range(ncv):
        atrf = np.mean(np.asarray(trfscv[i * ncv2:(i + 1) * ncv2]), axis=0)
        x1 = Xtest[i * ncv2]
        yactuals.append(Ytest[i * ncv2])
        yp1 = []
        yp1.append(x1 @ atrf)
        ypreds.append(np.asarray(yp1))
    ypred = np.concatenate(ypreds, axis=1)
    yactuals = np.concatenate(yactuals, axis=1)
    
    # compute test correlation
    corr = np.corrcoef(yactuals, ypred)[0, 1]        
        
    # compute average TRF across training folds
    avgamps = np.mean(np.asarray(ampscv), axis=0)
    avglats = np.mean(np.asarray(latscv), axis=0)
    avgtrf = np.sum(np.asarray([amp * basis[:, int(np.round(lat))] for amp, lat in zip(avgamps, avglats)]), axis=0)
    
    SPdict = dict(trfscv=trfscv, corrscv=corrscv, 
                  latscv=latscv, ampscv=ampscv, 
                  avglats=avglats, avgamps=avgamps,
                  ystds=ystds, xstds=xstds)

    return avgtrf, corr, SPdict



def EMSP_wrapper(preds, y, K, wins, signs, basis, max_iter=5, n_comp=3, par=False):
    '''
    wrapper for computing multichannel TRF using Expectation Maximization and Subspace Pursuit with cross-validation
    
    inputs
    - preds : list of predictor vector NDVars
    - y : multi channel measurements NDVar
    - K : length of TRF
    - wins : windows for TRF components
    - signs : signs of TRF components
    - basis : TRF basis dictionary
    - max_iter : maximum iterations for EMSP
    - n_comp : number of components in the TRF for each predictor
    - par : enable to compute using parallel processing
    
    outputs
    - avgtrf : average TRF across training folds
    - corr : correlation on test data
    - avgtrf_indep : average TRF across training folds using independent SP at each channel
    - corr_indep : correlation on test data using independent SP at each channel
    - EMSPdict : dictionary with fitted metrics, training TRFs and correlations
    - ystds : standardized measurements
    - xstds : standardized predictors
    '''
    
    importlib.reload(dataprep)
    
    # standardize predictor vectors and convert to matrix
    n_pred = len(preds)
    Xs = []
    xstds = []
    for ip in range(n_pred):
        x = preds[ip].x
        N = len(x)
        xstds.append(np.std(x))
        x = (x - np.mean(x))/np.std(x)
        X = dataprep.x_to_X(x, K, edges=0)
        Xs.append(X)
    X = np.concatenate(Xs, axis=1)

    # standardize measurements
    y = y.x
    M = y.shape[0]
    ymeans = []
    ystds = []
    for m in range(M):
        ymeans.append(np.mean(y[m]))
        ystds.append(np.std(y[m]))
        y[m] = (y[m] - np.mean(y[m])) / np.std(y[m])
    import pdb; pdb.set_trace()  
    # make data folds for crossvalidation
    ncv = 4 # number of cv folds
    ncv2 = 3 # number of inner cv folds
    Ytrain, Yvalid, Ytest, Xtrain, Xvalid, Xtest = dataprep.crossval_4fold(y, X)

    # initialize
    corrscv = []
    trfscv = []
    latscv = []
    ampscv = []
    toposcv = []
    corrscv_indep = [] # fit SP independently at each channel
    trfscv_indep = [] # fit SP independently at each channel
    latscv_indep = [] # fit SP independently at each channel
    ampscv_indep = [] # fit SP independently at each channel
    toposcv_indep = [] # fit SP independently at each channel
    J = n_comp * n_pred # 3 components in the TRF for each predictor
    
    # prepare function handle for parallel processing
    # leave cv as the only free parameter
    fEMSPcv = functools.partial(EMSPcv, Ytrain.copy(), Xtrain.copy(), Yvalid.copy(), Xvalid.copy(), wins,
                                 signs, basis, J, 0, max_iter)
    
   
    if par: # if parallel processing
        pool = multiprocessing.Pool(os.cpu_count())
        presult = list(pool.map(fEMSPcv, range(len(Ytrain))))
        for p in presult:
            trfscv.append(p[0])
            corrscv.append(p[1])
            latscv.append(p[2])
            ampscv.append(p[3])
            toposcv.append(p[4])
            trfscv_indep.append(p[5])
            corrscv_indep.append(p[6])
            latscv_indep.append(p[7])
            ampscv_indep.append(p[8])
    else:
        for cv in range(len(Ytrain)):
            trf, corr, lat, amp, spat, trf_indep, corr_indep, lat_indep, amp_indep = fEMSPcv(cv)
            trfscv.append(trf)
            corrscv.append(corr)
            latscv.append(lat)
            ampscv.append(amp)
            toposcv.append(spat)
            trfscv_indep.append(trf_indep)
            corrscv_indep.append(corr_indep)
            latscv_indep.append(lat_indep)
            ampscv_indep.append(amp_indep)
    
    # average fits over training folds
    avgamps = np.mean(np.asarray(ampscv), axis=0)
    avglats = np.mean(np.asarray(latscv), axis=0)
    avglats = np.asarray([int(np.round(ll)) for ll in avglats])
    avgtopos = np.mean(np.asarray(toposcv), axis=0)
    avgtrf = maketrf(basis, avgamps, avglats, avgtopos)

    # average fits from independent SP on each channel
    avgamps_indep = np.mean(np.asarray(ampscv_indep), axis=0)
    avglats_indep = np.mean(np.asarray(latscv_indep), axis=0)
    avgtrf_indep = []
    for m in range(M):
        avglats_indep2 = np.asarray([int(np.round(ll)) for ll in avglats_indep[m]])
        avgamps_indep2 = np.asarray(avgamps_indep[m])
        avgtrf_indep2 = maketrf(basis, avgamps_indep2, avglats_indep2, np.ones((6, 1)))
        avgtrf_indep.append(avgtrf_indep2[0])
    avgtrf_indep = np.asarray(avgtrf_indep)
    
    # correlation
    ypreds = []
    yactuals = []
    for i in range(ncv):
        atrf = np.mean(np.asarray(trfscv[i * ncv2:(i + 1) * ncv2]), axis=0)
        x1 = Xtest[i * ncv2]
        yactuals.append(Ytest[i * ncv2])
        yp1 = atrf @ x1.T
        ypreds.append(np.asarray(yp1))
    ypred = np.concatenate(ypreds, axis=-1)
    yactuals = np.concatenate(yactuals, axis=-1)
    corr = []
    for m in range(M):
        corr.append(np.corrcoef(yactuals[m], ypred[m])[0, 1])
    
    # correlation for independent fits at each channel
    ypreds = []
    yactuals = []
    for i in range(ncv):
        atrf = np.mean(np.asarray(trfscv_indep[i * ncv2:(i + 1) * ncv2]), axis=0)
        x1 = Xtest[i * ncv2]
        yactuals.append(Ytest[i * ncv2])
        yp1 = atrf @ x1.T
        ypreds.append(np.asarray(yp1))
    ypred = np.concatenate(ypreds, axis=-1)
    yactuals = np.concatenate(yactuals, axis=-1)
    corr_indep = []
    for m in range(M):
        corr_indep.append(np.corrcoef(yactuals[m], ypred[m])[0, 1])
    
    EMSPdict = dict(trfscv=trfscv, corrscv=corrscv, 
                    ampscv=ampscv, latscv=latscv, toposcv=toposcv, 
                    avgamps=avgamps, avglats=avglats, avgtopos=avgtopos,
                    ystds=ystds, xstds=xstds)
    
    SPdict = dict(trfscv_indep=trfscv_indep, corrscv_indep=corrscv_indep, 
                  ampscv_indep=ampscv_indep, latscv_indep=latscv_indep, 
                  avgamps_indep=avgamps_indep, avglats_indep=avglats_indep,
                  ystds=ystds, xstds=xstds)

    
    return avgtrf, corr, avgtrf_indep, corr_indep, EMSPdict, SPdict




def EMSPcv(Ytrain, Xtrain, Yvalid, Xvalid, wins, signs, basis, J, verbose, max_iter, cv):
    '''
    Inner function for EMSP cross validation. Always call from EMSPwrapper
    '''
    YTr = Ytrain[cv]
    XTr = Xtrain[cv]
    YVal = Yvalid[cv]
    XVal = Xvalid[cv]
    errorflag = True
    while errorflag:
        errorflag, trf, amps, lats, topos, corr, SPindep = EMSP(YTr, XTr, YVal, XVal, wins, signs, basis, J=J,
                                                                    max_iter=max_iter, verbose=verbose)
    return trf, corr, lats, amps, topos, SPindep['trfs'], SPindep['errs']['corr'], SPindep['lats'], SPindep['amps']




def EMSP(YTr, XTr, YVal, XVal, wins, signs, basis, J=3, max_iter=10, verbose=0):
    '''
    EMSP algorithm
    dimensions J = # components, M = channels, N = time, K = TRF lags (may be concatenated predictors)
    inputs:
    - YTr : (M x Ntr) training measurements
    - XTr : (Ntr x K) training predictors
    - YVal : (M x Nval) validation data
    - XVal : (Nval x K) validation predictors
    - wins : [[window_start window_end] for j in range(J)]
    - signs : [sign for j in range(J)]
    - basis : (K x K) TRF basis
    - J : # components
    - max_iter : EM iterations
    
    outputs:
    - error : boolean
    - trf :
    - amps :
    - lats :
    - topos :
    - corr :
    - SPindep : dictionary result from SP fit independently at each channel
    '''
    
    M, N = YTr.shape
    K = basis.shape[0]
    
    # placeholder values
    amps = np.ones(J)  # component amplitudes
    lats = np.zeros(J).astype(int) # component latencies
    topo_mu = np.zeros(M) # topography mean mu
    covR = np.eye(M) # topographhy covariance R
    covL = np.eye(M) # spatial covariance of noise Lambda
    
    # apply basis dictionary
    XbTr = XTr @ basis
    XbVal = XVal @ basis

    
    # Initialize at SP fit independently at each channel
    SPindep = SP_multichannel_independent(YTr, XTr, YVal, XVal,  basis, wins, J, None)
    topos = SPindep['amps'].T # initialize topographies to be amplitudes at each channel
    for j in range(J):
        lats[j] = int(np.mean(np.asarray(SPindep['lats'])[:, j])) # initialize latencies
    for j in range(J):
        amps[j] = np.max(np.abs(np.asarray(topos[j,:]))) # initialize amplitudes
        topos[j] /= amps[j] # normalize topographies
            
    Sjs = [np.eye(M)]*J

    # start with EMstep
    topos, amps, covR, covL, topo_mu = EMstep(YTr.copy(), XbTr.copy(), 
                                                       J, amps, lats, basis, topos, signs,  
                                                       covR=covR, covL=covL, topo_mu=topo_mu)

    # start EM iterations
    prev_err = np.inf
    prev_corr = -np.inf
    for i_iter in range(max_iter):
        amps, lats = SP_multichannel(YTr.copy(), XbTr.copy(), YVal.copy(), XbVal.copy(), J, wins,
                                         signs, basis, topos, amps, covL=covL)

        topos, amps, covR, covL, topo_mu = EMstep(YTr.copy(), XbTr.copy(), 
                                                       J, amps, lats, basis, topos, signs, 
                                                       covR=covR, covL=covL, topo_mu=topo_mu)

        save.pickle(topos, f'topos_{i_iter}.pkl')
        # compute correlation
        trf = maketrf(basis, amps, lats, topos)
        Ypred = trf @ XVal.T
        corr = np.mean([np.corrcoef(Ypred[m], YVal[m])[0, 1] for m in range(M)])

        if verbose > 0:
            print(f'end of iter {i_iter}, {np.mean(corr)}', ' ' * 20, end='\r')
            
        # compute error
        err1 = np.linalg.norm((YVal - Ypred).flatten())
        
        # stop iterations if correlation reduces
        if np.mean(corr) <= np.mean(prev_corr) and i_iter > 2:
            trf = prev_trf
            amps = prev_amps
            lats = prev_lats
            topos = prev_topos
            corr = prev_corr
            break
            
        prev_err = err1.copy()
        prev_corr = corr.copy()
        prev_trf = trf.copy()
        prev_amps = amps.copy()
        prev_lats = lats.copy()
        prev_topos = topos.copy()
        
    if verbose > 0:
        print(f'end {i_iter}, {np.mean(corr)}, {lats}, {amps}', ' ' * 20, end='\r')
        print(f'{np.mean(corr)}, {lats}, {amps}, {np.max(np.abs(topos), axis=1)}', ' ' * 20)
    return False, trf, amps, lats, topos, corr, SPindep




def SP_multichannel_independent(YTr, XTr, YVal, XVal, basis, wins, J, signs):
    '''
    fit SP independently at each channel
    '''
    
    M, N = YTr.shape
    K = XTr.shape[1]
    XbTr = XTr @ basis
    XbVal = XVal @ basis

    # initialize values
    amps = np.zeros((M, J))
    lats = np.zeros((M, J))
    trfs = np.zeros((M, K))
    corrs = np.zeros(M) # correlations
    l2s = np.zeros(M) # l2 errors
    l1s = np.zeros(M) # l1 errors
    
    for m in range(M):
        # run SP singlechannel
        amps1, lats1 = SP_singlechannel(YTr[m,:].copy(), XbTr.copy(), YVal[m,:].copy(), XbVal.copy(), J, wins, signs, amps)
        amps[m, :] = amps1
        lats[m, :] = lats1
        lats = lats.astype(int)
        
        # predicted signal
        trfs[m, :] = np.sum(np.asarray([amp * basis[:, abs(lat)] for amp, lat in zip(amps[m, :], lats[m, :])]), axis=0)
        Ypred = np.zeros(N)
        for amp, lat in zip(amps[m, :], lats[m, :]):
            Ypred += amp * XbTr[:, abs(lat)]
        
        # metrics
        corrs[m] = np.corrcoef(YTr[m, :], Ypred)[0, 1]
        l2s[m] = np.linalg.norm(YTr[m, :] - Ypred) ** 2
        l1s[m] = np.sum(np.asarray(np.abs(YTr[m, :] - Ypred)))
        
    # make result dictionary
    SPindep = {}
    SPindep['amps'] = amps
    SPindep['lats'] = abs(lats)
    SPindep['trfs'] = trfs
    SPindep['errs'] = {
        'corr': corrs,
        'l2': np.mean(l2s),
        'l1': np.mean(l1s),
    }
    return SPindep



def EMstep(Y_in, Xb, J, amps, lats, basis, topos, signs, covR=None, covL=None, topo_mu=None, covRflag=False):
    
    # initialize
    covLi = scipy.linalg.pinv(covL)
    covRi = scipy.linalg.pinv(covR)
    Y = Y_in.copy()
    M, N = Y.shape
    K = Xb.shape[1]
    topos_old = topos.copy()
    topos = np.zeros(topos_old.shape)
    amps_old = amps.copy()
    
    topo_mu = np.mean(topos_old, axis=0)

    
    Sjs = [np.eye(M)]*J
    lats = np.asarray(lats).astype(int)
    # calculate topographies in E-step
    for j in range(J):
        Sjs[j] = np.linalg.pinv((Xb[:, lats[j]].T @ Xb[:, lats[j]]) * covLi + covRi) 
        zeta = Y.copy()
        for j2 in range(J):
            if j2 == j:
                continue
            zeta -= ((Xb[:, lats[j2]])[:, np.newaxis] @ (amps[j2]*topos_old[j2:j2 + 1, :])).T
        sp_mu1 = np.zeros(M)
        sp_mu1 = covLi @ zeta @ Xb[:, lats[j]] + covRi @ topo_mu
        
        topos[j, :] = Sjs[j] @ sp_mu1

    # Conditional Maximization steps
    # update topography prior parameters
    topo_mu = np.mean(topos, axis=0)
    
    # update spatial covariance R
#     if covRflag:
#         covR1 = np.zeros((M,M))
#         for j in range(J):
#             covR1 += (Sjs[j] + topos[j,:][:,np.newaxis]@topos[j,:][np.newaxis,:]).T - 2*topo_mu@topos[j,:].T + topo_mu@topo_mu.T
#         covR = covR1 / (J*M)
    
    # update noise spatial covariance Lambda
    covL1 = Y@Y.T
    covL2 = np.zeros((M,N))
    for j in range(J):
        covL2 += topos[j,:][:,np.newaxis] @ Xb[:, lats[j]][np.newaxis,:]
    covL2 = Y @ covL2.T + covL2 @ Y.T
    covL3 = np.zeros((M, M))
    for j in range(J):
        covL3 += (Xb[:, lats[j]][np.newaxis,:] @ Xb[:, lats[j]][:,np.newaxis]) * (Sjs[j] + topos[j,:][:,np.newaxis]@topos[j,:][np.newaxis,:]).T
        for i in range(J):
            if i==j:
                continue
            covL3 += (Xb[:, lats[j]][np.newaxis,:] @ Xb[:, lats[i]][:,np.newaxis]) * (topos[j,:][:,np.newaxis]@topos[i,:][np.newaxis,:])
    covL = (covL1 - covL2 + covL3 + np.eye(M))/N
    
    # update component amplitudes and normalize topographies
    for j in range(J):
        if np.max(np.abs(topos[j, :])) == 0:
            amps[j] = 0
        else:
            amps[j] = np.max(np.abs(topos[j, :]))
            topos[j, :] /= amps[j]
        
    return topos, amps, covR, covL, topo_mu



def SP_singlechannel(y_in, Xb_in, yVal, XbVal, J, wins, signs, amps, max_iter=10):
    '''
    compute TRF at a single channel using Subspace Pursuit
    multiple predictors should be concatenated in Xb_in, XbVal and J = n_components * n_predictors
    inputs
    - y_in : measurement vector
    - Xb_in : Toeplitz predictor matrix. Note that TRF basis dictionary has already been applied
    - yVal : measurement vector of validation data
    - XbVal : predictor matrix of validation data (basis applied)
    - J : number of components
    - signs : list of sublists of windows for each component [[start_lag, end_lag] for j in range(J)]
    - amps :  amplitudes of components
    - max_iter : maximum iterations for SP algorithm
    
    outputs
    - amps : amplitudes of components
    - lats : latencies of components
    '''
    # initialize
    Xb = Xb_in.copy()
    N, K = Xb.shape
    y = y_in.copy()
    r = y.copy() # initialize residual
    lats = [0] * J
    prev_corr = -np.inf
    prev_err = np.inf
    
    # set of selected components. implemented as a list of J sublists of components
    setC = []
    for j in range(J):
        setC.append([])
        
    for i_iter in range(max_iter):
        for j in range(J): # iterate over windows
            if signs is not None: # include sign when computing dot product, else use the absolute value of dot product
                if signs[j] > 0:
                    obj = [np.dot(r, Xb[:, k]) for k in range(int(wins[j][0]), int(wins[j][1]))]
                else:
                    obj = [np.dot(r, -Xb[:, k]) for k in range(int(wins[j][0]), int(wins[j][1]))]
            else:
                obj = [np.abs(np.dot(r, Xb[:, k])) for k in range(int(wins[j][0]), int(wins[j][1]))]

            best_lat = int(np.argmax(obj)) + wins[j][0]
            # if best latency is not in setC[j], add it to setC[j]
            if best_lat not in setC[j]:
                setC[j].append(best_lat)
        
        A = [] # matrix with columns as Xc for each component c in setC
        for j in range(J):
            for lat in setC[j]:
                A.append(Xb[:, lat])
        A = np.asarray(A).T
        ampsA = np.linalg.pinv(A.T @ A) @ A.T @ y # includes previous selected and current selected components
        
        # select only the maximum amplitude components
        ampsAj = [] # amplitudes with sublists for each component
        ii = 0
        for j in range(J):
            ampsAj.append([])
            for k in setC[j]:
                ampsAj[j].append(ampsA[ii])
                ii += 1
            # select only this one
            max_i = np.argmax(np.abs(ampsAj[j])) # get max amp for each j
            # update setC and latencies
            setC[j] = [setC[j][max_i]]
            lats[j] = setC[j][0]
            
            
        B = [] # matrix with columns as Xc for each component c in setC
        for j in range(J):
            B.append(Xb[:, setC[j]][:,0])
        B = np.asarray(B).T
        amps = np.linalg.pinv(B.T @ B) @ B.T @ y # amplitudes for best components
        
        # predicted signal
        pred = np.zeros(y.shape)
        for j in range(J):
            pred += amps[j] * Xb[:, lats[j]]
            
        # compute error
        err = np.linalg.norm(y - pred)
        # update residual
        r = y - pred
        
        # predicted validation signal
        predVal = np.zeros(yVal.shape)
        for j in range(J):
            predVal += amps[j] * XbVal[:, lats[j]]
        
        # compute correlations
        corr = np.corrcoef(yVal, predVal)[0, 1]
        
        # stop iterations
        if err >= prev_err and i_iter > 2:
            amps = prev_amps
            lats = prev_lats
            break
        
        prev_amps = amps.copy()
        prev_lats = lats.copy()
        prev_err = err
        prev_corr = corr

    return amps, lats



def SP_multichannel(Y_in, Xb_in, YTe, XbTe, J, wins, signs, basis, topos_in, amps_in, covL=None):
    
    if covL is not None:
        covLi = scipy.linalg.pinv(covL)
        covLisqrt = np.real(scipy.linalg.sqrtm(covLi))
        Y = covLisqrt @ Y_in.copy()
        topos = topos_in.copy()
        for j in range(J):
            topos[j] = (covLisqrt @ topos_in[j][:,np.newaxis])[:,0]
    else:
        Y = Y_in.copy()
        topos = topos_in.copy()
        
    topos_uw = topos_in.copy() # unwhitened for predTe calculation
    
    Xb = Xb_in.copy()
    M, N = Y.shape
    N, K = Xb.shape
    max_iter = 10
    
    r = Y.copy()

    lats = [0] * J
    
    # list of J sublists of components
    selected = []
    for j in range(J):
        selected.append([])
        
    prev_corr = -np.inf
    prev_err = np.inf
    
    for i_iter in range(max_iter):
        for j in range(J): # iterate over windows
            obj = [np.abs(np.dot(r.flatten(), ((Xb[:, k:k + 1] @ topos[j][np.newaxis, :]).T).flatten())) 
                  for k in range(int(wins[j][0]), int(wins[j][1]))]
            
            # if best lat is not in selected[j], add it to selected[j]
            best_lat = int(np.argmax(obj) + wins[j][0])
            if best_lat not in selected[j]:
                selected[j].append(best_lat)

        D1 = []
        for j in range(J):
            for k in selected[j]:
                Xb1 = np.zeros(Y.shape)
                for m in range(M):
                    Xb1[m, :] = topos[j, m] * Xb[:, k]
                D1.append(Xb1.flatten())
        D1 = np.asarray(D1).T

        amps1 = np.linalg.pinv(D1.T @ D1) @ D1.T @ Y.flatten() # includes previous selected and current selected
        
        sel_amps = []
        ii = 0
        D2 = []
        for j in range(J):
            sel_amps.append([])
            for k in selected[j]:
                sel_amps[j].append(amps1[ii])
                ii += 1
            max_i = np.argmax(np.abs(sel_amps[j])) # get max amp in each j
            
            lats[j] = selected[j][max_i]
            selected[j] = [selected[j][max_i]]

            Xb1 = np.zeros(Y.shape)
            for m in range(M):
                Xb1[m, :] = topos[j, m] * Xb[:, lats[j]]
            D2.append(Xb1.flatten())
            
        D2 = np.asarray(D2).T
        amps = np.linalg.pinv(D2.T @ D2) @ D2.T @ Y.flatten()
                    
        for j in range(J):
            if amps[j] < 0:
                amps[j] *= -1
                topos[j] *= -1
                topos_uw[j] *= -1

        pred = np.zeros(Y.shape)
        for j in range(J):
            pred += amps[j] * (Xb[:, lats[j]:lats[j] + 1] @ topos[j][np.newaxis, :]).T
            
        r = Y - pred

        
        predTe = np.zeros(YTe.shape)
        for j in range(J):
            predTe += amps[j] * (XbTe[:, lats[j]:lats[j] + 1] @ topos_uw[j][np.newaxis, :]).T
        
        err = np.linalg.norm((Y - pred).flatten())

        corr = []
        for m in range(M):
            corr.append(np.corrcoef(YTe[m, :], predTe[m, :])[0, 1])
        
        if err >= prev_err and i_iter > 2:
            amps = prev_amps
            lats = prev_lats
            break
        
        prev_amps = amps.copy()
        prev_lats = lats.copy()
        prev_err = err
        prev_corr = np.mean(corr)

    return amps, lats

def maketrf(basis, amps, lats, topos):
    K, K1 = basis.shape
    J, M = topos.shape
    trf = np.zeros((M, K))
    for j in range(J):
        trf += topos[j:j + 1, :].T @ (amps[j] * basis[:, int(np.round(lats[j]))][:, np.newaxis]).T
    return trf
