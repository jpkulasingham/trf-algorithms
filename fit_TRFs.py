from eelbrain import *
import mne
import scipy
import numpy as np
import os, sys
import importlib
import ridge
import dataprep
import make_sim_data
import EMSP
import time


def fit_TRFs(ynd, predsnd, algs, params, fs, dim):
    importlib.reload(ridge)
    importlib.reload(EMSP)
    npred = len(predsnd)
    trfs1 = {}
    corrs1 = {}
    algdicts = {}
    eldict = {}
    
    # add edge samples
    tstart = params['boost']['tstart']
    tstop = params['boost']['tstop']
    params1 = {}
    for k in params['boost'].keys():
        params1[k] = params['boost'][k]
    params1['tstart'] = tstart - 0.1
    params1['tstop'] = tstop + 0.1
    tt = time.time()
    
    if 'boost' in algs:
        # 4-way crossvalidation
        if dim=='1D':
            ynd4 = combine([NDVar(ynd.sub(time=(i*45,(i+1)*45)), UTS(0,1/fs,int(45*fs))) for i in range(4)])
        else:
            ynd4 = combine([NDVar(ynd.sub(time=(i * 45, (i + 1) * 45)), (ynd.dims[0], UTS(0, 1/fs, int(45*fs)))) for i in range(4)])
        predsnd4 = []
        for pred in predsnd:
            predsnd4.append([NDVar(pred.sub(time=(i*45,(i+1)*45)), UTS(0,1/fs,int(45*fs))) for i in range(4)])
        print(ynd4)
        res = boosting(ynd4, predsnd4, **params1)
        elb = time.time() - tt
        if dim=='1D':
            corrs1['boost'] = res.r
        else:
            corrs1['boost'] = res.r.x.copy()
        trfs1['boost'] = []
        for ip in range(npred):
            trfs1['boost'].append(res.h_scaled[ip].sub(time=(tstart, tstop)).x.copy())
        print(f'boosting elapsed {int(np.floor(elb/60))}:{int(np.floor(elb%60))}, corr = {np.mean(corrs1["boost"]):.5f}')
        algdicts['boost'] = dict(res=res)
        eldict['boost'] = elb
        
        
    if 'ridge' in algs:
        print('ridge', ' ' * 20, end='\r')
        K = params['ridge']['K']
        tt = time.time()
        r_avgtrf, r_corr, r_res_dict = ridge.ridge_wrapper(predsnd.copy(), ynd.copy(), **params['ridge'], par=True)
        algdicts['ridge'] = r_res_dict
        el_r = time.time() - tt
        corrs1['ridge'] = np.asarray(r_corr)
        trfs1['ridge'] = []
        for ip in range(npred):
            if dim=='1D':
                trfs1['ridge'].append(r_avgtrf[0,ip * K:(ip + 1) * K] * r_res_dict['ystds'][0] / r_res_dict['xstds'][ip])
            else:
                trf = []
                for m in range(r_avgtrf.shape[0]):
                    trf.append(r_avgtrf[m,ip*K:(ip+1)*K]*r_res_dict['ystds'][m]/r_res_dict['xstds'][ip])
                trfs1['ridge'].append(np.asarray(trf))
            
        print(f'ridge elapsed {int(np.floor(el_r/60))}:{int(np.floor(el_r%60))}, corr = {np.mean(corrs1["ridge"]):.5f}')
        eldict['ridge'] = el_r
    
    if 'sp' in algs:
        print('SP', ' ' * 20, end='\r')
        K = params['sp']['K']  # tt = time.time()
        tt = time.time()
        if dim=='1D':
            sp_avgtrf, sp_corr, sp_dict = EMSP.SP_singlechannel_wrapper(predsnd.copy(), ynd.copy(), **params['sp'])
        else:
            emsp_avgtrf, emsp_corr, sp_avgtrf, sp_corr, emsp_dict, sp_dict = EMSP.EMSP_wrapper(predsnd.copy(), ynd.copy(), **params['sp'])

        algdicts['sp'] = sp_dict
        el_sp = time.time() - tt
        corrs1['sp'] = np.asarray(sp_corr)
        trfs1['sp'] = []
        
        if dim!='1D':
            algdicts['emsp'] = emsp_dict
            el_emsp = el_sp
            corrs1['emsp'] = np.asarray(emsp_corr)
            trfs1['emsp'] = []
        
        for ip in range(npred):
            if dim=='1D':
                trfs1['sp'].append(sp_avgtrf[ip * K:(ip + 1) * K] * sp_dict['ystds'][0] / sp_dict['xstds'][ip])
            else:
                trf = []
                trfem = []
                for m in range(sp_avgtrf.shape[0]):
                    trf.append(sp_avgtrf[m,ip*K:(ip+1)*K]*sp_dict['ystds'][m]/sp_dict['xstds'][ip])
                    trfem.append(emsp_avgtrf[m,ip*K:(ip+1)*K]*emsp_dict['ystds'][m]/emsp_dict['xstds'][ip])
                trfs1['sp'].append(np.asarray(trf))
                trfs1['emsp'].append(np.asarray(trfem))

        print(f'SP elapsed {int(np.floor(el_sp/60))}:{int(np.floor(el_sp%60))}, SP corr = {np.mean(corrs1["sp"]):.5f}')
        eldict['sp'] = el_sp
        
        if dim!='1D':
            print(f'EMSP elapsed {int(np.floor(el_emsp/60))}:{int(np.floor(el_emsp%60))}, EMSP corr = {np.mean(corrs1["emsp"]):.5f}')
            eldict['emsp'] = el_emsp

    return trfs1, corrs1, eldict, algdicts

  
