from eelbrain import *
import numpy as np
import scipy, os, importlib
import fit_simulations as fsim
import make_sim_data as mksim

root_folder = 'path_to_root_folder'
infolder = f'{root_folder}/test001'

importlib.reload(fsim)
importlib.reload(mksim)
sim_data = load.unpickle(f'{infolder}_data_source/sim_data_source.pkl')
outfolder = f'{infolder}_source'

rng = np.random.RandomState(2022)

force_make = False
for snr in [-15,-20,-25,-30]:
    print('SNR', snr)
    outfolder1 = f'{outfolder}/noise_snr{snr}'
    if not os.path.exists(outfolder1):
        os.makedirs(outfolder1)
    if os.path.exists(f'{outfolder1}/sim_data_source_snr{snr}.pkl') and not force_make:
        sim_data1 = load.unpickle(f'{outfolder1}/sim_data_source_snr{snr}.pkl')
    else:
        sim_data1 = mksim.sim_add_noise(root_folder, sim_data, snr, 'source', rng=rng)
        save.pickle(sim_data1, f'{outfolder1}/sim_data_source_snr{snr}.pkl')

    algs = ['boost', 'ridge', 'sp', 'emsp']
    n_loop = 1
    npred = 2
    fsim.loop_fit_TRFs_simdata(outfolder1, sim_data1, n_loop, npred, algs, dim='source', nperm=3, snr=snr, rng=rng, onlyperm=True)

