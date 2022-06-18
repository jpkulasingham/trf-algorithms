from eelbrain import *
import mne, scipy, time, os, sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import fit_TRFs as fTRF
import make_sim_data
import dataprep


importlib.reload(fTRF)
root_folder = 'path_to_root_folder'
dssdict = load.unpickle(f'{root_folder}/data/data_dss6dict.pkl')
sensordict = load.unpickle(f'{root_folder}/data/data_sensordict.pkl')
preds1 = load.unpickle(f'{root_folder}/preds/foreground_predictor.pkl')
preds1 = concatenate(preds1, 'time')
preds2 = load.unpickle(f'{root_folder}/preds/background_predictor.pkl')
preds2 = concatenate(preds2, 'time')

preds = [preds1, preds2]
npred = len(preds)

L = len(dssdict['ydss6A'])

algs = ['boost', 'ridge', 'sp', 'emsp']

fs = 100
wins1 = [[0.02,0.08],[0.09,0.17],[0.19,0.25]]
wins_concat = []
for ip in range(npred):
    wins_concat += [[int((w+ip*0.5)*fs) for w in win] for win in wins1]
print(wins_concat)

trf_basis = make_sim_data.make_basis(100, 50*npred, width=0.05, spacing=0.01)

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
        'alpha_all': [2**i for i in range(8,13,1)],
        'K': 50,
        'COV':None,
    },
    'sp': {
        'K': 50,
        'wins': wins_concat,
        'signs': [1,-1,1]*4,
        'basis': trf_basis,
    },         
}

trfsA = {}
corrsA = {}
for k in algs:
    trfsA[k] = []
    corrsA[k] = []
sensordims = []
outfolder = f'{root_folder}/test_001/sensor'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

logfile = f'{outfolder}/log.txt'
with open(logfile, 'w+') as f:
    f.write('realdata_sensor.py\n')
    f.write(outfolder)
    f.write('\n')

force_make = True
for l in range(L):
    tt = time.time()
    subj = sensordict['subjs'][l]
    if subj == 'R2336':
        continue
    if subj == 'R2411':
        continue
        
    preds1 = []
    for pred in preds:
        preds1.append(pred.sub(time=(0,60*3)))
        
    ydss = concatenate(dssdict['ydss6A'][l], 'time')
    ydss = ydss.sub(time=(0,60*3))
    
    trfs, corrs, eldict, algdicts = fTRF.fit_TRFs(ydss, preds1, algs, algparams, fs, dim='sensor')
    for k in algs:
        corrsA[k].append(corrs[k])
    el = time.time() - tt
    resdict = dict(trfs=trfs, corrs=corrs, eldict=eldict, algdicts=algdicts)
    save.pickle(resdict, f'{outfolder}/{subj}_resdict_dss.pkl')
    dataprep.print_fit_results(logfile, 'DSS', sensordict['subjs'][l], corrs, el, algdicts['ridge']['alpha_all'], algdicts['ridge']['best_alpha'])
    
    
    ysensor = concatenate(sensordict['megs'][l], 'time')
    ysensor = ysensor.sub(time=(0,60*3))
    
    trfs, corrs, eldict, algdicts = fTRF.fit_TRFs(ysensor, preds1, algs, algparams, fs, dim='sensor')
    for k in algs:
        corrsA[k].append(corrs[k])
    el = time.time() - tt
    resdict = dict(trfs=trfs, corrs=corrs, eldict=eldict, algdicts=algdicts)
    save.pickle(resdict, f'{outfolder}/{subj}_resdict_sensor.pkl')
    dataprep.print_fit_results(logfile, 'SENSOR', sensordict['subjs'][l], corrs, el, algdicts['ridge']['alpha_all'], algdicts['ridge']['best_alpha'])

    
    for ip in range(3):
        print(f'PERM {ip}')
        
        preds1perm = dataprep.permute_predictors_nd(preds1, ip)
        trfs, corrs, eldict, algdicts = fTRF.fit_TRFs(ydss, preds1perm, algs, algparams, fs, dim='sensor')
        for k in algs:
            corrsA[k].append(corrs[k])
        el = time.time() - tt
        resdict = dict(trfs=trfs, corrs=corrs, eldict=eldict, algdicts=algdicts)
        save.pickle(resdict, f'{outfolder}/{subj}_resdict_dss_p{ip+1}.pkl')
        dataprep.print_fit_results(logfile, f'DSS PERM {ip}', sensordict['subjs'][l], corrs, el, algdicts['ridge']['alpha_all'], algdicts['ridge']['best_alpha'])
        
        trfs, corrs, eldict, algdicts = fTRF.fit_TRFs(ysensor, preds1perm, algs, algparams, fs, dim='sensor')
        for k in algs:
            corrsA[k].append(corrs[k])
        el = time.time() - tt
        resdict = dict(trfs=trfs, corrs=corrs, eldict=eldict, algdicts=algdicts)
        save.pickle(resdict, f'{outfolder}/{subj}_resdict_sensor_p{ip+1}.pkl')
        dataprep.print_fit_results(logfile, f'SENSOR PERM {ip}', sensordict['subjs'][l], corrs, el, algdicts['ridge']['alpha_all'], algdicts['ridge']['best_alpha'])
