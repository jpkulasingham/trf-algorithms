from eelbrain import *
import numpy as np
import scipy, os, importlib

import fit_simulations as fsim
import make_sim_data as mksim

root_folder = 'path_to_root_folder'
infolder = f'{root_folder}/test_001'

importlib.reload(fsim)
importlib.reload(mksim)
sim_data = load.unpickle(f'{infolder}_data_1D/sim_data_1D.pkl')
outfolder = f'{infolder}_1D'

rng = np.random.RandomState(2022)

force_make = False
for snr in [-15,-20,-25,-30]:
    print('SNR', snr)
    outfolder1 = f'{outfolder}/noise_snr{snr}'
    if not os.path.exists(outfolder1):
        os.makedirs(outfolder1)
    sim_data1 = sim_data
    algs = ['boost', 'ridge', 'sp']
    n_loop = 5
    npred = 2
    fsim.loop_fit_TRFs_simdata(outfolder1, root_folder, sim_data1, n_loop, npred, algs, dim='1D', nperm=3, snr=snr, rng=rng, onlyperm=False)

