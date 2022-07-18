from eelbrain import *
import mne, time, os, importlib, dataprep
import scipy, scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt


def sim_TRFs(root_folder, outfolder, subjects_dir=None, dim='1D', L=30, fs=100, rng=None):
    '''
    simulate 1-dimensional responses and trfs
    inputs:
    - root_folder: path to data folder
    - outfolder: path to output folder
    - subject_dir: path to source space MRIs
    - dim: type of data to simulate. Can be '1D', 'sensor' or 'source'
    - L: number of subjects
    - fs: sampling frequency
    - npred: number of predictors
    '''
    if rng==None:
        rng = np.random.RandomState(2022)

    Lh = int(L / 2)
    group = [0] * Lh + [1] * Lh # divide into two groups (e.g., older and younger)
    K = int(fs / 2) # TRF lags (500 ms)
    J = 3 # number of components
    
    print('loading predictors')
    fg = load.unpickle(f'{root_folder}/preds/foreground_predictor.pkl') # 2 predictors
    bg = load.unpickle(f'{root_folder}/preds/background_predictor.pkl') # 2 predictors
    preds = [fg, bg]
    npred = len(preds)
    
    # convert predictor vectors to matrices
    Xs = []
    for pred in preds:
        Xs.append(dataprep.x_to_X(pred.x, K)) 
        
    # component ground truths
    wins = [[[0.03, 0.08], [0.09, 0.17], [0.19, 0.25]] for _ in range(npred)] # component windows for each predictor (in seconds)
    pklats = [[0.05, -0.12, 0.2], [0.07, -0.15, 0.23]] # component latencies for each predictor (in seconds)
    pkamps = [[0.8, 1, 0.5], [0.8, 1, 0.5]] # component amplitudes for each predictor    
#     ampG1 = [np.asarray([4, 2, 4]), np.asarray([4, 0.3, 4])] # change in amplitudes for group 1
#     ampG2 = [np.asarray([0.5, 2, -0.2]), np.asarray([1, 0.1, -0.2])] # change in amplitudes for group 2
    ampG1 = [np.asarray([1, 2, 1]), np.asarray([1, 0.3, 1])] # change in amplitudes for group 1
    ampG2 = [np.asarray([0.5, 2, -0.2]), np.asarray([1, 0.1, -0.2])] # change in amplitudes for group 2
    tauG1 = [[0.02, 0, 0.02], [0, 0.03, 0.03]] # change in latencies for group 1 (in seconds)
    tauG2 = [[0, 0, 0], [0, 0, 0]] # change in latencies for group 2 (in seconds)
    signs = [[1, -1, 1], [1, -1, 1]] # component signs
    
    print('making ground truth trfs')
    gt_lats, gt_amps = add_individual_variability(pklats, pkamps, [ampG1, ampG2], [tauG1, tauG2], L, J, fs, wins, outfolder=outfolder, rng=rng)

    btrf = make_basis(fs, K) # basis for TRF
    gt_trfs = []
    for l in range(L):
        gt_trfs1 = []
        for ip in range(npred):
            gt_trfs1.append(make_trf(gt_lats[ip][l], gt_amps[ip][l], signs[ip], btrf))
        gt_trfs.append(gt_trfs1)
    gt_trfs = np.asarray(gt_trfs)

    print('simulating clean data')
    if dim == '1D':
        Yc, Yc_pred = sim_indiv_clean_1D(gt_trfs, Xs)
        signs = [[1, -1, 1]] * npred
        sim_data = dict(gt_trfs=gt_trfs, gt_lats=gt_lats, gt_amps=gt_amps, signs=signs, outfolder=outfolder, 
                     wins=wins, btrf=btrf, Yc=Yc, Yc_pred=Yc_pred, Xs=Xs, fs=fs)
    else:
        gt_trfs_1D = gt_trfs.copy()
        
        if dim == 'sensor':
            gt_spatial, sensordim = sim_topos(root_folder, outfolder, L, J, npred, rng)
            spatialdict = dict(spatialdim=sensordim)
        elif dim == 'source':
            spatialdict = sim_source_distributions(root_folder, subjects_dir, L, npred)
            gt_spatial = spatialdict['src_dist_bp']
        
        M = len(spatialdict['spatialdim'])
        gt_trfs = np.zeros((L, npred, M, K))
        for l in range(L):
            for ip in range(npred):
                gt_trfs[l, ip, :, :] = make_trf_MD(gt_lats[ip][l], gt_amps[ip][l], gt_spatial[ip][l], signs[ip], btrf)
        gt_trfs = np.asarray(gt_trfs)
        
        Yc, Yc_pred = sim_indiv_clean_MD(gt_trfs, Xs)
        
        sim_data = dict(gt_trfs=gt_trfs, gt_trfs_1D=gt_trfs_1D, gt_lats=gt_lats, gt_amps=gt_amps, signs=signs, outfolder=outfolder,
                        btrf=btrf, Yc=Yc, Yc_pred=Yc_pred, Xs=Xs, fs=fs,
                        spatialdim=spatialdict['spatialdim'], gt_spatial=gt_spatial, spatialdict=spatialdict)
        
    save.pickle(sim_data, f'{outfolder}/sim_data_{dim}.pkl')

    return sim_data



def add_individual_variability(pklats, pkamps, ampGs, tauGs, L, J, fs, wins, jit=0.02, outfolder=None, rng=None):
    Lh = int(L / 2)
    npred = len(pklats)
    gt_lats = [] # ground truth latencies
    gt_amps = [] # ground truth amplitudes
    
    # convert seconds to samples
    jit = int(jit*fs)    
    
    for ip in range(npred):
        # random initialization
        gt_amps1 = 0.5 * np.abs(rng.randn(L, J))
        gt_lats1 = rng.randint(-jit, jit, size=(L, J)) # latency shifts
        
        # add group changes
        gt_amps1[:Lh, :] += ampGs[0][ip]
        gt_amps1[Lh:, :] += ampGs[1][ip]
        
        gt_lats1[:Lh, :] += np.asarray([int(t*fs) for t in tauGs[0][ip]])
        gt_lats1[Lh:, :] += np.asarray([int(t*fs) for t in tauGs[1][ip]])

        for l in range(L):
            # add ground truth latencies and amplitudes
            gt_amps1[l, :] += pkamps[ip]
            gt_lats1[l, :] += np.round([abs(int(lat * fs)) for lat in pklats[ip]])

            # ensure latencies are within component windows
            for j in range(J):
                if gt_lats1[l, j] < wins[ip][j][0]*fs + 1:
                    gt_lats1[l, j] = wins[ip][j][0]*fs + 1
                if gt_lats1[l, j] > wins[ip][j][1]*fs - 1:
                    gt_lats1[l, j] = wins[ip][j][1]*fs - 1
                    
        gt_lats1 = gt_lats1.astype(int)
        
        gt_lats.append(gt_lats1.copy())
        gt_amps.append(gt_amps1.copy())
        
    # plot latencies and amplitudes
    if outfolder is not None:
        plt.figure(figsize=(20, 10))
        for j in range(3):
            for ip in range(npred):
                plt.subplot(3, npred, j * npred + ip + 1)
                plt.hist(np.asarray(gt_lats)[ip, :, j])
        plt.savefig(f'{outfolder}/gt_latencies.jpg', bbox_inches='tight')

        plt.figure(figsize=(20, 10))
        for j in range(3):
            for ip in range(npred):
                plt.subplot(3, npred, j * npred + ip + 1)
                plt.hist(np.asarray(gt_amps)[ip, :, j])
        plt.savefig(f'{outfolder}/gt_amplitudes.jpg', bbox_inches='tight')

    return gt_lats, gt_amps


def sim_indiv_clean_1D(gt_trfs, Xs):
    L = len(gt_trfs)
    npred = len(gt_trfs[0])
    N = len(Xs[0])
    Yc = []
    Yc_preds = []
    for l in range(L):
        Yc1 = np.zeros(N)
        Yc_preds1 = []
        for ip in range(npred):
            Yc_11 = Xs[ip] @ gt_trfs[l][ip].T
            Yc_preds1.append(Yc_11)
            Yc1 += Yc_11
        Yc.append(Yc1)
        Yc_preds.append(Yc_preds1)
    return Yc, Yc_preds


def sim_indiv_clean_MD(t_trfs, Xs):
    L = len(t_trfs)
    npred = len(t_trfs[0])
    N = len(Xs[0])
    M = t_trfs[0][0].shape[0]
    Yc = []
    Yc_preds = []
    for l in range(L):
        Yc1 = np.zeros((M, N))
        Yc_preds1 = []
        for ip in range(npred):
            Yc_11 = (Xs[ip] @ t_trfs[l][ip].T).T
            Yc_preds1.append(Yc_11)
            Yc1 += Yc_11
        Yc.append(Yc1)
        Yc_preds.append(Yc_preds1)
    return Yc, Yc_preds


def sim_topos(root_folder, outfolder, L, J, npred, rng):
    if not os.path.exists(f'{outfolder}/sim_topos'):
        os.makedirs(f'{outfolder}/sim_topos')
    topos_in = load.unpickle(f'{root_folder}/gt_sensor_topos.pkl')
#     for j in range(J):
#         topos_in[j].x[0] = 0
    topos = np.zeros((npred, L, J, len(topos_in[0].sensor)))
    for l in range(L):
        for ip in range(npred):
            plotnds = []
            for j in range(J):
                topos[ip, l, j, :] = topos_in[j].x + 0.1 * rng.randn(len(topos_in[j].sensor))
                plotnds.append(NDVar(topos[ip, l, j, :], topos_in[j].sensor))
            p = plot.Topomap(plotnds, ncol=3)
            p.save(f'{outfolder}/sim_topos/topo_l{l}_ip{ip}.png')
            p.close()
    return topos, topos_in[0].sensor

def sim_source_distributions(root_folder, subjects_dir, L, npred):
    src_comps = load.unpickle(f'{root_folder}/source_distributions/source_components.pkl')
    aparcLabels = ['superiortemporal', 'transversetemporal']
    aparcLabels = [l+'-lh' for l in aparcLabels] + [l+'-rh' for l in aparcLabels]

    J = 3
    src_comp_in = []
    src_comp_in.append(src_comps['hg'])
    src_comp_in.append(-src_comps['pt']) # (-) for M100
    src_comp_in.append(src_comps['stg'])

    src_dist = np.zeros((npred, L, J, len(src_comp_in[0].source)))
    src_distbp = np.zeros(src_dist.shape)
    src_comp_bp = np.zeros((L, J, len(src_comp_in[0].source)))
    src_dist_bp = np.zeros((npred, L, J, len(src_comp_in[0].source)))
    sensor_comp = []
    for l in range(L):
        print(f'source_comp {l}', ' '*20, end='\r')
        sens1c = []
        fwd = mne.read_forward_solution(f'{root_folder}/source_distributions/fwd_{l:02d}-fwd.fif')
        invdict = load.unpickle(f'{root_folder}/source_distributions/invdict_{l:02d}.pkl')
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
        fwd = load.fiff.forward_operator(fwd, invdict['src_space'], subjects_dir=subjects_dir, parc='aparc')
        fwd = fwd.sub(source=aparcLabels)
        invdict['info1']['meas_date'] = None

        for j in range(J):
            sens1 = ss_to_sens(src_comp_in[j], fwd)
            sens1c.append(sens1)
            ss1 = sens_to_ss(sens1, invdict, aparcLabels, subjects_dir)
            src_comp_bp[l,j,:] = ss1.x
            for ip in range(npred):
                src_dist[ip, l, j, :] = src_comp_in[j].x + 0.05*rng.randn(len(src_comp_in[j].source))
                sens1 = ss_to_sens(NDVar(src_dist[ip,l,j,:], src_comp_in[j].source), fwd)
                ss1 = sens_to_ss(sens1, invdict, aparcLabels, subjects_dir) 
                src_dist_bp[ip, l, j, :] = ss1.x
        sensor_comp.append(sens1c)
        
    spatialdict = dict(src_comp=src_comp_in, src_comp_bp=src_comp_bp, sensor_comp=sensor_comp, 
                       src_dist=src_dist, src_dist_bp=src_dist_bp, spatialdim=src_comp_in[0].source)
    return spatialdict

def ss_to_sens(ss, fwd):
    return NDVar(np.dot(fwd.x, ss.x), fwd.sensor)

def sens_to_ss(sens, invdict, aparcLabels, subjects_dir, method='MNE', lambda2=0.111, fixed=True):
    invsol = invdict['invsol']
    subject = invdict['subject']
    src_space = invdict['src_space']
    ep = mne.EpochsArray(sens.x[np.newaxis,:,np.newaxis], invdict['info1'])
    stc = mne.minimum_norm.apply_inverse_epochs(ep, invsol, lambda2=1, method='MNE', pick_ori=None)
    snd = load.fiff.stc_ndvar(stc, subject, src_space, subjects_dir)
    snd = snd[0].sub(source=aparcLabels).sub(time=0)
    snd = morph_source_space(snd, 'fsaverage')
    return snd

def make_basis(fs, K, width=0.07, spacing=0.01):
    spacingN = int(spacing * fs)
    nbasis = int(K / spacingN)
    wN = int(width * fs)
    wNh = int(wN / 2)
    hamm = scipy.signal.windows.hamming(wN)
    hN = len(hamm)
    hNh = int(hN / 2)
    btrf = np.zeros((K, nbasis))
    for bb in range(1, nbasis):
        c = bb * spacingN
        if c < hNh:
            btrf[:, bb] = np.concatenate([hamm[hNh - c:], np.zeros(K - hNh - c - 1)])
        elif c >= K - hNh:
            btrf[:, bb] = np.concatenate([np.zeros(c - hNh), hamm[:hNh + (K - c)]])
        else:
            btrf[:, bb] = np.concatenate([np.zeros(c - hNh), hamm, np.zeros(K - c - hN + hNh)])
    K1 = btrf.shape[1]
    return btrf

def make_trf(pkidxs, pkamps, signs, basis):
    K, K1 = basis.shape
    trf = np.zeros((K, 1))[:, 0]
    for pkidx, pkamp, sign in zip(pkidxs, pkamps, signs):
        pkidx1 = int((K1 / K) * pkidx)
        trf += sign * pkamp * basis[:, pkidx1]
    return trf


def make_trf_MD(pkidxs, pkamps, pktopos, signs, basis):
    K, K1 = basis.shape
    M = len(pktopos[0])
    trf = np.zeros((M, K))
    for pkidx, pkamp, pktopo in zip(pkidxs, pkamps, pktopos):
        pkidx1 = int((K1 / K) * pkidx)
        trf += pktopo[:, np.newaxis] @ (pkamp * basis[:, pkidx1])[np.newaxis, :]
    return trf


def sim_add_noise(root_folder, sim_data, snr, dim, jitter=0, dropout=0, fs=100, rng=None):
    jitdicts = None

    if jitter>0 or dropout>0:
        if dim=='1D':
            Yjit, Yjit_pred, jitdicts = sim_indiv_jitter_1D(sim_data['gt_lats'], sim_data['gt_amps'], sim_data['Xs'], 
                                                     jitter, dropout, sim_data['signs'], sim_data['btrf'], sim_data['wins'], rng=rng)
        else:
            Yjit, Yjit_pred, jitdicts = sim_indiv_jitter_MD(sim_data['gt_lats'], sim_data['gt_amps'], sim_data['gt_spatial'], sim_data['Xs'], 
                                                     jitter, dropout, sim_data['signs'], sim_data['btrf'], sim_data['wins'], rng=rng)
    else:
        Yjit, Yjit_pred = sim_data['Yc'].copy(), sim_data['Yc_pred'].copy()
    
    Yjit = np.asarray(Yjit)
    Yjit_pred = np.asarray(Yjit_pred)

    print('add noise')
    if dim=='1D':
        Y, megnoise = noise_meg_1D(root_folder, Yjit.copy(), snr, rng=rng)
    else:
        Y, megnoise = noise_meg_MD(root_folder, Yjit.copy(), snr, sim_data['spatialdim'], dim, rng=rng)

    sim_data1 = {}
    for k in sim_data.keys():
        if k in ['Yc', 'Yc_pred']:
            continue
        if isinstance(sim_data[k], (list, np.ndarray)):
            sim_data1[k] = sim_data[k].copy()
        else:
            sim_data1[k] = sim_data[k]
    sim_data1['Y'], sim_data1['Yjit'], sim_data1['Yjit_pred'], sim_data1['jitdicts'] = Y, Yjit, Yjit_pred, jitdicts
    sim_data1['snr'], sim_data1['megnoise'] = snr, megnoise
    
    return sim_data1


def noise_meg_1D(root_folder, Yc, snr, fs=100, sgap=500, rng=None):
    Y = np.zeros(np.asarray(Yc).shape)
    N = Y.shape[1]
    megnoiseA = []
    megpath = f'{root_folder}/megnoise/noise_1DSS.pkl'
    meg = load.unpickle(megpath)
    meg = concatenate(meg, 'time').x
    for l in range(Y.shape[0]):
        megnoise = shuffle_phase(meg, rng)
        megnoise = shrink_outliers(megnoise, 7)
        megnoise = megnoise[:N+sgap]
        megnoise = filter_data(NDVar(megnoise, UTS(0, 1 / fs, len(megnoise))), 1, 10).x[sgap:]
        Y[l, :], _, _ = add_noise_snr_1D(Yc[l, :], megnoise, snr, fs)
        megnoiseA.append(megnoise)
    return Y, megnoiseA


def noise_meg_MD(root_folder, Yc, snr, spatialdim, dim, fs=100, sgap=500, rng=None):
    Y = np.zeros(np.asarray(Yc).shape)
    L, M, N = Y.shape
    megnoiseA = np.zeros((L, M, N))
    if dim=='sensor':
        megpath = f'{root_folder}/megnoise/noise_sensor.pkl'
        meg = load.unpickle(megpath)
    elif dim=='source':
        megpath = f'{root_folder}/megnoise/noise_source.pkl'
        ds = load.unpickle(megpath)
        meg = concatenate(ds['source'], 'time')
        spatialdim.subjects_dir =  f'{root_folder}/source_distributions/mri'
        spatialdim.subject = 'fsaverage'
    megx = meg.x
    for l in range(Y.shape[0]):
        print(l, end='\r')
        megnoise = np.zeros((M, N+sgap))
        for m in range(M):
            megnoise1 = shuffle_phase(megx[m].copy(), rng)
            megnoise1 = shrink_outliers(megnoise1, 7)
            megnoise1 = megnoise1[:N+sgap]
            megnoise[m, :] = megnoise1.copy()
        megnd = NDVar(megnoise, (spatialdim, UTS(0, 1 / fs, N+sgap)))
        megnd = megnd.smooth(dim, 0.02, 'gaussian')
        megnoise = filter_data(megnd, 1, 10).x[:,sgap:]
        Y[l, :, :], _, _ = add_noise_snr_MD(Yc[l, :, :], megnoise, snr, fs)
        megnoiseA[l, :, :] = megnoise
    return Y, megnoiseA


def shuffle_phase(signal, rng):
    Xf = scipy.fftpack.fft(signal)
    Xfm = np.abs(Xf) ** 2
    Xfp = np.angle(Xf)
    plh = Xfp[1:int(len(Xfp) / 2)]
    rng.shuffle(plh)
    prh = -plh[::-1]
    Xfp2 = np.append(Xfp[0], np.append(plh, np.append(Xfp[int(len(Xfp) / 2)], prh)))
    Xf2 = np.sqrt(Xfm) * np.exp(1j * Xfp2)
    signal = np.real(scipy.fftpack.ifft(Xf2))
    return signal

def shrink_outliers(data, m=2):
    outliers = abs(data - np.mean(data)) > m * np.std(data)
    data[outliers] = np.sign(data[outliers]) * m * np.std(data)
    return data

def add_noise_snr_1D(signal, noise, snr, fs):
    sP = np.sum(np.abs(np.fft.fft(signal, fs // 2) / len(signal)) ** 2)
    nP = np.sum(np.abs(np.fft.fft(noise, fs // 2) / len(signal)) ** 2)
    snrfactor = 10 ** (snr / 10)
    noise *= np.sqrt(sP / (snrfactor * nP))
    sP = np.sum(np.abs(np.fft.fft(signal, fs // 2) / len(signal)) ** 2)
    nP = np.sum(np.abs(np.fft.fft(noise, fs // 2) / len(signal)) ** 2)
    snr1 = 10 * np.log10(sP / nP)
    signal_noise = signal + noise
    return signal_noise, signal, noise

def add_noise_snr_MD(signal, noise, snr, fs):
    M, N = signal.shape
    signal_noise = np.zeros(signal.shape)
    noise_snr = np.zeros(noise.shape)
    for m in range(M):
        signal_noise[m, :], _, noise_snr[m, :] = add_noise_snr_1D(signal[m, :], noise[m, :], snr, fs)
    return signal_noise, signal, noise
