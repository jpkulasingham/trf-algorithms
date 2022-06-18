from eelbrain import *
import mne, importlib, ridge, scipy, os, sys, dataprep, EMSP, time
import numpy as np
import make_sim_data as mksim
import fit_TRFs as fTRF

def fit_TRFs_simdata(sim_data, outfolder, logfile, npred, algs, dim, L = 30, rng=None, snr=None, iperm=0):
    time_s = time.time()
    fs = sim_data['fs']
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    importlib.reload(ridge)
    importlib.reload(EMSP)
    importlib.reload(fTRF)
    
    # make predictor ndvars
    preds = []
    predsnd = []
    for ip in range(npred):
        pred = sim_data['Xs'][ip][:,0].copy()
        preds.append(pred)
        predsnd.append(NDVar(pred, UTS(0,0.01,len(pred))))
    
     
    # component windows
    wins = [[0.03, 0.08], [0.09, 0.17], [0.19, 0.25]]
    wins_concat = [] # concatenated windows for each predictor
    for ip in range(npred):
        wins_concat += [[int((w+ip*0.5)*fs) for w in win] for win in wins]
        
    trf_basis = mksim.make_basis(fs, int(0.5*fs*npred), width=0.05, spacing=0.01)
    
    if dim=='1D':
        alpha_all = [2**i for i in range(-8,10,2)]
    elif dim=='sensor':
        alpha_all = [2 ** i for i in range(-4, 10, 2)]
    elif dim=='source':
        alpha_all = [2 ** i for i in range(0, 14, 2)]
        
    alpha_all = [2**i for i in range(0,8,1)]
    
    algs_gt = ['gt'] + algs
    
    algparams = {
        'boost': {
            'tstart': 0,
            'tstop': 0.5,
            'partitions': 4,
            'test': 1,
            'basis':0.05,
            'selective_stopping': True,
            'partition_results': True,
        },
        'ridge': {
            'alpha_all': alpha_all,
            'K': sim_data['Xs'][0].shape[1],
            'COV':None,
        },
        'sp': {
            'K': sim_data['Xs'][0].shape[1],
            'wins': wins_concat,
            'signs': [1,-1,1]*4,
            'basis': trf_basis,
        },         
    }
    
    tt = time.time()
    for l in range(L):
        trfs = {}
        corrs = {}
        
        time_sl = time.time()
        y = sim_data['Y'][l].copy()
        if dim=='1D':
            N = len(y)
            ynd = NDVar(y, UTS(0,1/fs,N))
        else:
            M, N = y.shape
            ynd = NDVar(y, (sim_data['spatialdim'], UTS(0, 1/fs, N)))
         
        if dim=='1D':
            ygt = np.zeros(N)
            for ip in range(npred):
                ygt += (sim_data['Xs'][ip] @ sim_data['gt_trfs'][l][ip].T).copy()
            trfs['gt'] = sim_data['gt_trfs'][l].copy()
            corrs['gt'] = np.corrcoef(y, ygt)[0,1]
        else:
            ygt = np.zeros((M, N))
            for ip in range(npred):
                ygt += (sim_data['Xs'][ip] @ sim_data['gt_trfs'][l][ip].T).T.copy()
            trfs['gt'] = sim_data['gt_trfs'][l].copy()
            corr = []
            for m in range(M):
                corr.append(np.corrcoef(y[m], ygt[m])[0, 1])
            corrs['gt'] = corr
        
        if dim=='sensor':
            y_for_dss = combine([NDVar(ynd.sub(time=(60*ii,60*(ii+1))).x,(sim_data['spatialdim'], UTS(0,1/fs,60*100))) for ii in range(3)])
            todss, fromdss = dss(y_for_dss)
            ydssnd = ynd.dot(todss[:6])
            ydssnd = NDVar(ydssnd.x.T, (ydssnd.dss, ydssnd.time))
            datadict = dict(y=y, ynd=ynd, todss=todss, fromdss=fromdss, ydssnd=ydssnd)
            save.pickle(datadict, f'{outfolder}/subj{l:02d}_pred{npred}_data.pkl')
            
            ygt = np.zeros((6, N))
            for ip in range(npred):
                gt_trfnd = NDVar(sim_data['gt_trfs'][l][ip], (sim_data['spatialdim'], UTS(0,1/fs,50)))
                gt_trfdss = gt_trfnd.dot(todss[:6]).x.T
                ygt += (sim_data['Xs'][ip] @ gt_trfdss.T).T.copy()
            trfs['gtdss'] = sim_data['gt_trfs'][l].copy()
            corr = []
            for m in range(6):
                corr.append(np.corrcoef(ydssnd.x[m], ygt[m])[0, 1])
            corrs['gtdss'] = corr
            
            trfsdss1, corrsdss1, eldictdss, algdictsdss = fTRF.fit_TRFs(ydssnd, predsnd, algs, algparams, fs, dim)
            for k in trfsdss1.keys():
                trf1 = []
                for ip in range(npred):
                    trf1.append(NDVar(trfsdss1[k][ip], (ydssnd.dss, UTS(0,1/fs,50))).dot(fromdss[:,:6]).x.T)
                trfs[k+'dss'] = trf1
                corrs[k + 'dss'] = corrsdss1[k]

            el = time.time() - tt
            dataprep.print_fit_results(logfile, f'SNR {snr} DSS PERM {iperm}', f'SUBJECT {l}', corrs, el, 
                                       algdictsdss['ridge']['alpha_all'], algdictsdss['ridge']['best_alpha'])
            
        trfs1, corrs1, eldict, algdicts = fTRF.fit_TRFs(ynd, predsnd, algs, algparams, fs, dim)
 
        for k in trfs1.keys():
            trfs[k] = trfs1[k].copy()
            corrs[k] = corrs1[k].copy()
            print(k, np.mean(corrs[k]))
            
        if dim=='sensor':
            for k in algdictsdss.keys():
                algdicts[k+'dss'] = algdictsdss[k]

        el = time.time() - time_s
        res = dict(trfs=trfs, corrs=corrs, eldict=eldict, el=el, algdicts=algdicts)
        if iperm>0:
            save.pickle(res, f'{outfolder}/subj{l:02d}_pred{npred}_res_p{iperm}_sp_emsp.pkl')
        else:
            save.pickle(res, f'{outfolder}/subj{l:02d}_pred{npred}_res_sp_emsp.pkl')

        
        el = time.time() - tt
        dataprep.print_fit_results(logfile, f'SNR {snr} {dim} PERM {iperm}', f'SUBJECT {l}', corrs, el, 
                                       algdicts['ridge']['alpha_all'], algdicts['ridge']['best_alpha'])

        

def loop_fit_TRFs_simdata(outfolder, root_folder, sim_data, n_loops, npred, algs, dim='1D', L=30, nperm=0, snr=None, rng=None, onlyperm=False):
    importlib.reload(mksim)

    for i in range(n_loops):
        print(f'\nSIM FIT LOOP {i}\n')
        outfolder1 = f'{outfolder}/sim_{i:03d}'
        if not os.path.exists(outfolder1):
            os.makedirs(outfolder1)
        logfile = f'{outfolder1}/log.txt'
        
        if not onlyperm:
            with open(logfile, 'w+') as f:
                f.write(f'{outfolder1}, n_loops = {n_loops}, {dim}\n')

            if n_loops > 1 and dim=='1D':
                sim_data1 = mksim.sim_add_noise(root_folder, sim_data, snr, dim, rng=rng)
                save.pickle(sim_data1, f'{outfolder1}/sim_data_1D_{snr}.pkl')
            else:
                sim_data1 = sim_data
            fit_TRFs_simdata(sim_data1, outfolder1, logfile, dim=dim, L=L, npred=npred, algs=algs, rng=rng, snr=snr, iperm=0)
    
        for iperm in range(nperm):
            print(f'\nSIM FIT PERM {iperm+1} LOOP {i}\n')
            if n_loops > 1 and dim=='1D':
                sim_data1 = load.unpickle(f'{outfolder1}/sim_data_1D_{snr}.pkl')
            else:
                sim_data1 = sim_data
            Xperms = dataprep.permute_predictors_Xs(sim_data1['Xs'], iperm)
            sim_data1['Xs'] = Xperms
            fit_TRFs_simdata(sim_data1, outfolder1, logfile, dim=dim, L=L, npred=npred, algs=algs, rng=rng, snr=snr, iperm=iperm+1)
