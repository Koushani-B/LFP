# -*- coding: utf-8 -*-
"""
LFP Analysis Pipeline for Multi-Region Electrophysiology
Matching Halverson, Kim & Freeman (2023, J Neurosci) methodology

Uses:
  - Complex Morlet wavelet (MNE-Python) for time-frequency decomposition
  - Per-trial pre-CS baseline z-normalization
  - One-way ANOVA + LSD post-hoc across 4 learning phases (UP/IT/LRN/RET)
  - Iterative outlier trimming (mean +/- 2 SD)

SETUP:
  pip install -r requirements.txt
  Update ANIMAL_CONFIGS below, then run:
  python koushani_lfp_analysis_vscode.py
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import pandas as pd
import os
import pickle
import warnings
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter
from mne.time_frequency import tfr_array_morlet
import pingouin as pg

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def save_figure(fig, filename, dpi=300):
    """Save figure to FIGURES_DIR as PNG and close."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filepath}")


def save_cache(data, name, animal_id):
    """Save preprocessed data to pickle cache."""
    filepath = os.path.join(CACHE_DIR, f'{animal_id}_{name}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"    Cached: {filepath} ({size_mb:.1f} MB)")


def load_cache(name, animal_id):
    """Load preprocessed data from pickle cache. Returns None if not found."""
    filepath = os.path.join(CACHE_DIR, f'{animal_id}_{name}.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"    Loaded from cache: {filepath} ({size_mb:.1f} MB)")
        return data
    return None


# ============================================================
# Data I/O Functions (unchanged)
# ============================================================

def read_single_ncs(filepath):
    """Read Neuralynx .ncs file"""
    dt = np.dtype([('timestamp', np.uint64), ('channel', np.uint32),
                   ('sample_rate', np.uint32), ('n_valid', np.uint32),
                   ('samples', np.int16, 512)])

    with open(filepath, 'rb') as f:
        header = f.read(16384).decode('latin-1')
        fs = float([line.split()[-1] for line in header.split('\r\n')
                   if '-SamplingFrequency' in line][0])
        data = np.fromfile(f, dtype=dt)

    lfp = data['samples'].flatten()
    return {
        'lfp': lfp,
        'times': np.arange(len(lfp)) / fs,
        'fs': fs,
        'channel': os.path.basename(filepath).replace('.ncs', '')
    }


def load_multiple_sessions(base_path, session_folders, channels_to_load=None):
    """Load LFP data from multiple sessions.

    Parameters
    ----------
    base_path : str
    session_folders : list or dict
    channels_to_load : set or list, optional
        If provided, only load these channel names (e.g., {'CSC5', 'CSC7', 'CSC14'}).
        This drastically reduces memory usage.
    """
    session_dict = session_folders if isinstance(session_folders, dict) else {f: f for f in session_folders}
    all_data = []

    for session_id, folder_name in session_dict.items():
        session_path = os.path.join(base_path, folder_name)
        if not os.path.exists(session_path):
            print(f"  Skipping: {session_path}")
            continue

        print(f"\n  Loading: {session_id}")
        ncs_files = sorted([f for f in os.listdir(session_path) if f.endswith('.ncs')])

        for ncs_file in ncs_files:
            # Filter to only requested channels
            if channels_to_load is not None:
                channel_name = ncs_file.replace('.ncs', '')
                if channel_name not in channels_to_load:
                    continue

            try:
                data = read_single_ncs(os.path.join(session_path, ncs_file))
                df = pd.DataFrame({
                    'time': data['times'], 'lfp': data['lfp'],
                    'channel': data['channel'], 'session': session_id, 'fs': data['fs']
                })
                all_data.append(df)
                print(f"    {data['channel']}: {len(data['lfp']):,} samples @ {data['fs']:.0f} Hz")
            except Exception as e:
                print(f"    {ncs_file}: {e}")

    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        print(f"\n  Loaded {df_all['session'].nunique()} sessions, "
              f"{df_all['channel'].nunique()} channels")
        return df_all
    return pd.DataFrame()


def read_neuralynx_events(filepath):
    """Read event file (.nev)"""
    dt = np.dtype([('stx', np.int16), ('pkt_id', np.int16), ('pkt_data_size', np.int16),
                   ('timestamp', np.uint64), ('event_id', np.int16), ('ttl', np.int16),
                   ('crc', np.int16), ('dummy1', np.int16), ('dummy2', np.int16),
                   ('extra', np.int32, 8), ('event_string', 'S128')])

    with open(filepath, 'rb') as f:
        f.read(16384)
        events = np.fromfile(f, dtype=dt)

    df = pd.DataFrame({'timestamp': events['timestamp'], 'ttl': events['ttl']})
    if len(df) > 0:
        df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e6
    return df


def read_all_events(session_path):
    """Read all .nev files from session"""
    nev_files = sorted([f for f in os.listdir(session_path) if f.endswith('.nev')])
    if not nev_files:
        return pd.DataFrame()

    all_events = []
    for nev_file in nev_files:
        try:
            df = read_neuralynx_events(os.path.join(session_path, nev_file))
            all_events.append(df)
            print(f"    {nev_file}: {len(df)} events")
        except Exception as e:
            print(f"    {nev_file}: {e}")

    return pd.concat(all_events, ignore_index=True).sort_values(
        'timestamp').reset_index(drop=True) if all_events else pd.DataFrame()


# ============================================================
# Trial Structure Functions (unchanged)
# ============================================================

def create_trial_structure(events_df, cs_ttl=8, us_ttl=16, skip_first=1):
    """Create trials from CS and US events"""
    cs_events = events_df[events_df['ttl'] == cs_ttl].sort_values('time_seconds').reset_index(drop=True)
    us_events = events_df[events_df['ttl'] == us_ttl].sort_values('time_seconds').reset_index(drop=True)

    print(f"    Total CS events: {len(cs_events)}, US events: {len(us_events)}")

    if skip_first > 0:
        cs_events = cs_events.iloc[skip_first:].reset_index(drop=True)
        us_events = us_events.iloc[skip_first:].reset_index(drop=True)
        print(f"    After skipping first {skip_first}: "
              f"CS = {len(cs_events)}, US = {len(us_events)}")

    trials = []
    for i in range(len(cs_events)):
        cs_time = cs_events.loc[i, 'time_seconds']
        is_cs_alone = ((i + 1) % 10 == 0)

        if is_cs_alone:
            us_time = np.nan
        else:
            us_cand = us_events[(us_events['time_seconds'] > cs_time) &
                               (us_events['time_seconds'] <= cs_time + 0.8)]
            us_time = us_cand.iloc[0]['time_seconds'] if len(us_cand) > 0 else np.nan

        trials.append({
            'trial_number': i + 1,
            'trial_type': 'cs_alone' if is_cs_alone else 'paired',
            'cs_onset_time': cs_time,
            'us_onset_time': us_time,
            'trial_start': cs_time - 0.2,
            'trial_end': (us_time + 0.5) if not np.isnan(us_time) else (cs_time + 1.5)
        })

    return pd.DataFrame(trials)


def extract_trial_lfp(df_all, trials_df, channel, session_id, time_window=(-0.2, 1.0)):
    """Extract LFP segments aligned to CS onset for each trial"""
    channel_data = df_all[
        (df_all['channel'] == channel) &
        (df_all['session'] == session_id)
    ].sort_values('time')

    if channel_data.empty:
        return []

    fs = channel_data['fs'].iloc[0]
    trial_data = []

    for _, trial in trials_df.iterrows():
        cs_time = trial['cs_onset_time']
        start_time = cs_time + time_window[0]
        end_time = cs_time + time_window[1]

        trial_segment = channel_data[
            (channel_data['time'] >= start_time) &
            (channel_data['time'] <= end_time)
        ]

        if len(trial_segment) > 0:
            time_rel = trial_segment['time'].values - cs_time
            trial_data.append({
                'lfp': trial_segment['lfp'].values,
                'time': time_rel,
                'fs': fs,
                'trial_number': trial['trial_number'],
                'trial_type': trial['trial_type'],
                'session': session_id,
            })

    return trial_data


# ============================================================
# Raw LFP Visualization (unchanged)
# ============================================================

def plot_raw_lfp_sample(df_all, session_id, regions, duration_sec=10,
                        start_time=0, figsize=(16, 8)):
    """Plot raw LFP traces from all regions"""
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=figsize, sharex=True)
    if n_regions == 1:
        axes = [axes]

    region_colors = {'Cerebellum': '#003f5c', 'Hippocampus': '#bc5090', 'ACC': '#ffa600'}

    for idx, (region_name, channels) in enumerate(regions.items()):
        ax = axes[idx]
        channel = channels[0]
        lfp_data = df_all[(df_all['session'] == session_id) &
                         (df_all['channel'] == channel)].sort_values('time')

        if not lfp_data.empty:
            time_mask = ((lfp_data['time'] >= start_time) &
                        (lfp_data['time'] <= start_time + duration_sec))
            segment = lfp_data[time_mask]

            if not segment.empty:
                ax.plot(segment['time'].values, segment['lfp'].values,
                       color=region_colors.get(region_name, 'black'),
                       linewidth=0.5, alpha=0.8)
                ax.set_ylabel(f'{region_name}\n({channel})',
                            fontsize=12, fontweight='bold',
                            color=region_colors.get(region_name, 'black'))
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                mean_lfp = np.mean(segment['lfp'].values)
                std_lfp = np.std(segment['lfp'].values)
                ax.text(0.02, 0.95, f'Mean={mean_lfp:.1f}\nSD={std_lfp:.1f}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, style='italic', color='red')

    axes[-1].set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    fig.suptitle(f'Raw LFP Traces - {session_id}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    safe_name = session_id.replace(' ', '_')
    save_figure(fig, f'raw_lfp_{safe_name}.png')


# ============================================================
# Morlet Wavelet Core Functions (NEW)
# ============================================================

def build_frequency_axis(freq_bands, n_bins_per_band=30):
    """
    Build frequency vector with ~30 bins per band (matching paper).
    Returns sorted unique frequencies and boolean masks per band.
    """
    all_freqs = []
    for band_name, (lo, hi) in freq_bands.items():
        band_freqs = np.linspace(lo, hi, n_bins_per_band)
        all_freqs.append(band_freqs)

    freqs = np.unique(np.concatenate(all_freqs))
    freqs.sort()

    band_masks = {}
    for band_name, (lo, hi) in freq_bands.items():
        band_masks[band_name] = (freqs >= lo) & (freqs <= hi)

    return freqs, band_masks


def compute_adaptive_n_cycles(freqs, min_cycles=3, scaling_factor=0.5):
    """
    Frequency-dependent cycle count for Morlet wavelets.
    Low freqs: fewer cycles (better time resolution).
    High freqs: more cycles (better frequency resolution).
    Paper guideline: >= 2-3 cycles minimum per bin.
    """
    return np.maximum(min_cycles, freqs * scaling_factor)


def compute_morlet_tfr(lfp_segments, fs, freqs, n_cycles):
    """
    Compute Morlet wavelet time-frequency power using MNE.

    Parameters
    ----------
    lfp_segments : list of 1D np.ndarray
        Each element is one trial's LFP voltage trace.
    fs : float
        Sampling frequency in Hz.
    freqs : np.ndarray
        Frequencies of interest.
    n_cycles : np.ndarray or float
        Wavelet cycles per frequency.

    Returns
    -------
    power : np.ndarray, shape (n_trials, n_freqs, n_times)
    """
    # Ensure equal-length segments
    min_len = min(len(seg) for seg in lfp_segments)
    data = np.array([seg[:min_len].astype(np.float64) for seg in lfp_segments])
    # MNE expects shape (n_epochs, n_channels, n_times)
    data = data[:, np.newaxis, :]

    tfr = tfr_array_morlet(
        data,
        sfreq=fs,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power',
        zero_mean=True,
        use_fft=True,
        decim=1,
        n_jobs=1,
        verbose=False
    )

    # Squeeze channel dim -> (n_trials, n_freqs, n_times)
    return tfr[:, 0, :, :]


def compute_morlet_complex_tfr(lfp_segments, fs, freqs, n_cycles):
    """
    Compute Morlet wavelet COMPLEX coefficients using MNE.
    Used for coherence analysis (cross-spectral density).

    Parameters
    ----------
    lfp_segments : list of 1D np.ndarray
        Each element is one trial's LFP voltage trace.
    fs : float
        Sampling frequency in Hz.
    freqs : np.ndarray
        Frequencies of interest.
    n_cycles : np.ndarray or float
        Wavelet cycles per frequency.

    Returns
    -------
    complex_tfr : np.ndarray, shape (n_trials, n_freqs, n_times)
        Complex Morlet wavelet coefficients (complex128).
    """
    min_len = min(len(seg) for seg in lfp_segments)
    data = np.array([seg[:min_len].astype(np.float64) for seg in lfp_segments])
    data = data[:, np.newaxis, :]  # (n_epochs, 1_channel, n_times)

    tfr = tfr_array_morlet(
        data,
        sfreq=fs,
        freqs=freqs,
        n_cycles=n_cycles,
        output='complex',
        zero_mean=True,
        use_fft=True,
        decim=1,
        n_jobs=1,
        verbose=False
    )

    # Squeeze channel dim -> (n_trials, n_freqs, n_times)
    return tfr[:, 0, :, :]


# ============================================================
# Coherence Core Functions (Phase 2: LFP-LFP Spectral Coherence)
# ============================================================

def compute_baseline_normalized_coherence(trial_lfps_x, trial_lfps_y, fs,
                                            freqs, n_cycles, baseline_window,
                                            analysis_window, trial_window,
                                            band_masks):
    """
    Compute coherence between two regions and z-normalize against
    pre-CS baseline. Returns both trial-averaged coherogram (for plots)
    and per-trial band coherence (for ANOVA).

    Parameters
    ----------
    trial_lfps_x : list of dict
        Trial LFP data for region X (from extract_trial_lfp).
    trial_lfps_y : list of dict
        Trial LFP data for region Y (same trials).
    fs : float
        Sampling frequency.
    freqs : np.ndarray
        Frequency vector.
    n_cycles : np.ndarray
        Wavelet cycles per frequency.
    baseline_window : tuple (start, end) in seconds
    analysis_window : tuple (start, end) in seconds
    trial_window : tuple (start, end) in seconds
    band_masks : dict {band_name: bool array}

    Returns
    -------
    per_trial_band_coh : dict {band_name: np.ndarray (n_trials,)}
        Per-trial z-scored coherence per band (for ANOVA).
    per_trial_band_icoh : dict {band_name: np.ndarray (n_trials,)}
        Per-trial z-scored imaginary coherence per band.
    z_coh_analysis : np.ndarray (n_freqs, n_analysis_times)
        Trial-averaged z-scored coherence TFR (for coherograms).
    z_icoh_analysis : np.ndarray (n_freqs, n_analysis_times)
        Trial-averaged z-scored imaginary coherence TFR.
    analysis_times : np.ndarray
    """
    # Match trial count
    n_trials = min(len(trial_lfps_x), len(trial_lfps_y))
    if n_trials == 0:
        empty = np.array([])
        return {}, {}, empty, empty, empty

    segments_x = [trial_lfps_x[i]['lfp'] for i in range(n_trials)]
    segments_y = [trial_lfps_y[i]['lfp'] for i in range(n_trials)]

    # Compute complex Morlet TFR for both regions
    complex_x = compute_morlet_complex_tfr(segments_x, fs, freqs, n_cycles)
    complex_y = compute_morlet_complex_tfr(segments_y, fs, freqs, n_cycles)
    # Each: (n_trials, n_freqs, n_times)

    # Build time vector and masks
    n_times = complex_x.shape[2]
    times = np.linspace(trial_window[0], trial_window[1], n_times)
    bl_mask = (times >= baseline_window[0]) & (times < baseline_window[1])
    an_mask = (times >= analysis_window[0]) & (times <= analysis_window[1])

    if bl_mask.sum() == 0 or an_mask.sum() == 0:
        print("    WARNING: Empty baseline or analysis window for coherence")
        empty = np.array([])
        return {}, {}, empty, empty, empty

    analysis_times = times[an_mask]

    # ---- Trial-averaged coherogram (for visualization) ----
    S_xy_avg = np.mean(complex_x * np.conj(complex_y), axis=0)  # (n_freqs, n_times)
    S_xx_avg = np.mean(np.abs(complex_x) ** 2, axis=0)
    S_yy_avg = np.mean(np.abs(complex_y) ** 2, axis=0)

    denom_avg = np.sqrt(S_xx_avg * S_yy_avg)
    denom_avg = np.where(denom_avg == 0, 1.0, denom_avg)

    coh_full = np.abs(S_xy_avg) / denom_avg       # (n_freqs, n_times)
    icoh_full = np.imag(S_xy_avg) / denom_avg

    # Z-normalize coherence against baseline per frequency
    bl_coh = coh_full[:, bl_mask]                   # (n_freqs, n_bl_times)
    bl_mean_coh = bl_coh.mean(axis=1, keepdims=True)  # (n_freqs, 1)
    bl_std_coh = bl_coh.std(axis=1, keepdims=True)
    bl_std_coh = np.where(bl_std_coh == 0, 1.0, bl_std_coh)

    z_coh_full = (coh_full - bl_mean_coh) / bl_std_coh
    z_icoh_full = (icoh_full - bl_mean_coh) / bl_std_coh

    z_coh_analysis = z_coh_full[:, an_mask]    # (n_freqs, n_analysis_times)
    z_icoh_analysis = z_icoh_full[:, an_mask]

    # ---- Per-trial band coherence (for ANOVA) ----
    # Average cross/auto spectra over analysis time AND band freqs BEFORE ratio
    # to avoid trivial Coh=1 at single time-freq points.
    per_trial_band_coh = {}
    per_trial_band_icoh = {}

    for band_name, mask in band_masks.items():
        trial_coh_vals = np.zeros(n_trials)
        trial_icoh_vals = np.zeros(n_trials)

        for i in range(n_trials):
            # Cross/auto spectra for this trial, band freqs, analysis time
            xy_i = complex_x[i, :, :] * np.conj(complex_y[i, :, :])
            xx_i = np.abs(complex_x[i, :, :]) ** 2
            yy_i = np.abs(complex_y[i, :, :]) ** 2

            # Average over band frequencies AND analysis time
            xy_band_mean = xy_i[mask, :][:, an_mask].mean()   # complex scalar
            xx_band_mean = xx_i[mask, :][:, an_mask].mean()   # real scalar
            yy_band_mean = yy_i[mask, :][:, an_mask].mean()   # real scalar

            denom_i = np.sqrt(xx_band_mean * yy_band_mean)
            if denom_i == 0:
                denom_i = 1.0

            trial_coh_vals[i] = np.abs(xy_band_mean) / denom_i
            trial_icoh_vals[i] = np.imag(xy_band_mean) / denom_i

        # Z-normalize per-trial coherence against baseline stats
        bl_band_coh = coh_full[mask, :][:, bl_mask].mean(axis=0)  # (n_bl_times,)
        bl_band_mean = bl_band_coh.mean()
        bl_band_std = bl_band_coh.std()
        if bl_band_std == 0:
            bl_band_std = 1.0

        per_trial_band_coh[band_name] = (trial_coh_vals - bl_band_mean) / bl_band_std
        per_trial_band_icoh[band_name] = (trial_icoh_vals - bl_band_mean) / bl_band_std

    # Free complex arrays to save memory
    del complex_x, complex_y

    return (per_trial_band_coh, per_trial_band_icoh,
            z_coh_analysis, z_icoh_analysis, analysis_times)


def compute_baseline_normalized_power(trial_lfps, fs, freqs, n_cycles,
                                       baseline_window, analysis_window,
                                       trial_window):
    """
    Compute Morlet TFR and z-normalize each trial against its own
    pre-CS baseline, per frequency bin.

    Parameters
    ----------
    trial_lfps : list of dict
        From extract_trial_lfp(). Each has 'lfp', 'time', 'fs', etc.
    fs : float
        Sampling frequency.
    freqs : np.ndarray
        Frequency vector.
    n_cycles : np.ndarray
        Cycles per frequency.
    baseline_window : tuple (start_sec, end_sec)
        e.g., (-0.2, 0.0) relative to CS onset.
    analysis_window : tuple (start_sec, end_sec)
        e.g., (0.0, 1.0) relative to CS onset.
    trial_window : tuple (start_sec, end_sec)
        Full extraction window, e.g., (-1.0, 1.5).

    Returns
    -------
    z_power_per_trial : np.ndarray, shape (n_trials, n_freqs)
        Mean z-scored power in analysis window per trial per freq.
    z_power_timecourse : np.ndarray, shape (n_trials, n_freqs, n_analysis_times)
        Full z-scored TFR in analysis window.
    analysis_times : np.ndarray
        Time vector for analysis window (seconds).
    """
    if not trial_lfps:
        return np.array([]), np.array([]), np.array([])

    # Compute Morlet TFR
    lfp_segments = [t['lfp'] for t in trial_lfps]
    power = compute_morlet_tfr(lfp_segments, fs, freqs, n_cycles)
    # power shape: (n_trials, n_freqs, n_times)

    # Build time vector matching TFR output
    n_times = power.shape[2]
    times = np.linspace(trial_window[0], trial_window[1], n_times)

    # Baseline and analysis time masks
    bl_mask = (times >= baseline_window[0]) & (times < baseline_window[1])
    an_mask = (times >= analysis_window[0]) & (times <= analysis_window[1])

    if bl_mask.sum() == 0 or an_mask.sum() == 0:
        print("    WARNING: Empty baseline or analysis window")
        return np.array([]), np.array([]), np.array([])

    analysis_times = times[an_mask]

    # Z-normalize per frequency using POOLED baseline across all trials.
    # For each frequency: compute mean and std from all baseline timepoints
    # across all trials, then z-score every timepoint in every trial.
    bl_power = power[:, :, bl_mask]  # (n_trials, n_freqs, n_bl_times)

    # Reshape to (n_freqs, n_trials * n_bl_times) to pool across trials
    n_trials_local = bl_power.shape[0]
    n_bl_times = bl_power.shape[2]
    bl_pooled = bl_power.transpose(1, 0, 2).reshape(len(freqs), -1)  # (n_freqs, n_trials*n_bl_times)

    bl_mean = bl_pooled.mean(axis=1)[np.newaxis, :, np.newaxis]  # (1, n_freqs, 1)
    bl_std = bl_pooled.std(axis=1)[np.newaxis, :, np.newaxis]    # (1, n_freqs, 1)
    bl_std = np.where(bl_std == 0, 1.0, bl_std)

    z_power_full = (power - bl_mean) / bl_std

    # Extract analysis window
    z_power_analysis = z_power_full[:, :, an_mask]

    # Mean z-power per trial per frequency (for band-power stats)
    z_power_per_trial = z_power_analysis.mean(axis=2)  # (n_trials, n_freqs)

    return z_power_per_trial, z_power_analysis, analysis_times


# ============================================================
# Data Collection Across Phases (NEW)
# ============================================================

def collect_phase_data(animal_id, config, freq_bands, freqs, n_cycles,
                       band_masks, trial_window, baseline_window,
                       analysis_window):
    """
    For one animal: load all sessions across all phases, compute
    per-trial baseline-normalized band power.

    Baseline is POOLED across ALL trials within a phase/region/channel
    (across sessions) before z-normalization.

    Returns
    -------
    results_df : pd.DataFrame
        Long-format: [animal, phase, region, channel, band,
                      trial_number, session, z_power]
    tfr_cache : dict
        phase -> region -> channel -> {
            'z_power_avg': array, 'analysis_times': array,
            'freqs': array
        }
        Averaged across sessions within phase. For spectrogram plotting.
    tfr_cache_csalone : dict
        Same structure but averaged over CS-alone (probe) trials only.
    """
    base_path = config['base_path']
    regions = config['regions']
    phases = config['phases']

    # Collect ALL session folders across phases
    all_sessions = []
    for phase_sessions in phases.values():
        all_sessions.extend(phase_sessions)

    # Build set of channels we actually need (saves ~80% memory)
    needed_channels = set()
    for channel_list in regions.values():
        needed_channels.update(channel_list)

    # Load only needed channels
    print(f"\n  Loading all sessions for {animal_id} "
          f"(channels: {needed_channels})...")
    df_all = load_multiple_sessions(base_path, all_sessions,
                                     channels_to_load=needed_channels)

    if df_all.empty:
        print(f"  ERROR: No data loaded for {animal_id}")
        return pd.DataFrame(), {}, {}

    # PASS 1: Accumulate trial LFPs per phase/region/channel across sessions
    # so we can pool baseline across ALL trials before z-normalization.
    trial_lfps_accum = {}  # (phase, region, channel) -> list of trial dicts

    for phase_name, session_list in phases.items():
        print(f"\n  Phase {phase_name}: {session_list}")

        for session_id in session_list:
            session_path = os.path.join(base_path, session_id)
            events_df = read_all_events(session_path)

            if events_df.empty:
                print(f"    WARNING: No events for {session_id}")
                continue

            trials_df = create_trial_structure(
                events_df,
                cs_ttl=config['cs_ttl'],
                us_ttl=config['us_ttl'],
                skip_first=config['skip_first']
            )

            if trials_df.empty:
                print(f"    WARNING: No trials for {session_id}")
                continue

            print(f"    {session_id}: {len(trials_df)} trials")

            for region_name, channels in regions.items():
                for channel in channels:
                    trial_lfps = extract_trial_lfp(
                        df_all, trials_df, channel, session_id,
                        time_window=trial_window
                    )

                    if not trial_lfps:
                        continue

                    key = (phase_name, region_name, channel)
                    if key not in trial_lfps_accum:
                        trial_lfps_accum[key] = []
                    trial_lfps_accum[key].extend(trial_lfps)

    # PASS 2: Compute TFR with pooled baseline for each group
    rows = []
    tfr_cache = {}
    tfr_cache_csalone = {}

    for (phase_name, region_name, channel), trial_lfps in \
            trial_lfps_accum.items():

        if phase_name not in tfr_cache:
            tfr_cache[phase_name] = {}
        if region_name not in tfr_cache[phase_name]:
            tfr_cache[phase_name][region_name] = {}
        if phase_name not in tfr_cache_csalone:
            tfr_cache_csalone[phase_name] = {}
        if region_name not in tfr_cache_csalone[phase_name]:
            tfr_cache_csalone[phase_name][region_name] = {}

        fs = trial_lfps[0]['fs']
        print(f"    {phase_name}/{region_name}/{channel}: "
              f"{len(trial_lfps)} trials (pooled), fs={fs:.0f} Hz")

        # Baseline is now pooled across ALL trials (all sessions in phase)
        z_per_trial, z_analysis, an_times = \
            compute_baseline_normalized_power(
                trial_lfps, fs, freqs, n_cycles,
                baseline_window, analysis_window, trial_window
            )

        if z_per_trial.size == 0:
            continue

        # Cache all-trial TFR average
        tfr_cache[phase_name][region_name][channel] = {
            'z_power_avg': z_analysis.mean(axis=0),  # (n_freqs, n_times)
            'n_trials': z_analysis.shape[0],
            'analysis_times': an_times,
            'freqs': freqs,
        }

        # CS-alone TFR cache
        csalone_indices = [i for i, t in enumerate(trial_lfps)
                           if t['trial_type'] == 'cs_alone']
        if csalone_indices:
            csalone_z = z_analysis[csalone_indices]
            tfr_cache_csalone[phase_name][region_name][channel] = {
                'z_power_avg': csalone_z.mean(axis=0),
                'n_trials': csalone_z.shape[0],
                'analysis_times': an_times,
                'freqs': freqs,
            }

        # Build long-format rows
        for trial_idx in range(z_per_trial.shape[0]):
            t_info = trial_lfps[trial_idx]
            t_num = t_info['trial_number']
            # Recover session from trial data (added below if not present)
            sess = t_info.get('session', 'unknown')

            for band_name, mask in band_masks.items():
                band_z = z_per_trial[trial_idx, mask].mean()

                rows.append({
                    'animal': animal_id,
                    'phase': phase_name,
                    'region': region_name,
                    'channel': channel,
                    'band': band_name,
                    'trial_number': t_num,
                    'session': sess,
                    'z_power': band_z,
                })

    results_df = pd.DataFrame(rows)
    return results_df, tfr_cache, tfr_cache_csalone


# ============================================================
# Coherence Data Collection Across Phases (Phase 2)
# ============================================================

def collect_coherence_data(animal_id, config, freq_bands, freqs, n_cycles,
                            band_masks, trial_window, baseline_window,
                            analysis_window, region_pairs):
    """
    For one animal: load all sessions, compute per-trial coherence
    for each region pair across all phases.

    Returns
    -------
    coh_results_df : pd.DataFrame
        Long-format: [animal, phase, pair, band, trial_number, session,
                      z_coherence, z_icoh]
    coh_cache : dict
        pair_label -> phase -> {z_coh_avg, z_icoh_avg, analysis_times, ...}
        Coherograms from ALL trials.
    coh_cache_csalone : dict
        Same structure but coherograms from CS-alone (probe) trials only.
    """
    base_path = config['base_path']
    regions = config['regions']
    phases = config['phases']

    # Collect ALL session folders across phases
    all_sessions = []
    for phase_sessions in phases.values():
        all_sessions.extend(phase_sessions)

    # Build set of channels needed across all region pairs
    needed_channels = set()
    for r1, r2 in region_pairs:
        needed_channels.update(regions[r1])
        needed_channels.update(regions[r2])

    # Load only needed channels
    print(f"\n  Loading sessions for coherence ({animal_id}, "
          f"channels: {needed_channels})...")
    df_all = load_multiple_sessions(base_path, all_sessions,
                                     channels_to_load=needed_channels)

    if df_all.empty:
        print(f"  ERROR: No data loaded for {animal_id}")
        return pd.DataFrame(), {}, {}

    rows = []
    coh_cache = {}
    coh_cache_csalone = {}

    # Initialize cache structure
    for r1, r2 in region_pairs:
        pair_label = f'{r1}-{r2}'
        coh_cache[pair_label] = {}
        coh_cache_csalone[pair_label] = {}

    for phase_name, session_list in phases.items():
        print(f"\n  Coherence - Phase {phase_name}: {session_list}")

        for session_id in session_list:
            session_path = os.path.join(base_path, session_id)
            events_df = read_all_events(session_path)

            if events_df.empty:
                print(f"    WARNING: No events for {session_id}")
                continue

            trials_df = create_trial_structure(
                events_df,
                cs_ttl=config['cs_ttl'],
                us_ttl=config['us_ttl'],
                skip_first=config['skip_first']
            )

            if trials_df.empty:
                print(f"    WARNING: No trials for {session_id}")
                continue

            print(f"    {len(trials_df)} trials")

            for (region_x, region_y) in region_pairs:
                pair_label = f'{region_x}-{region_y}'
                channel_x = regions[region_x][0]
                channel_y = regions[region_y][0]

                trial_lfps_x = extract_trial_lfp(
                    df_all, trials_df, channel_x, session_id,
                    time_window=trial_window
                )
                trial_lfps_y = extract_trial_lfp(
                    df_all, trials_df, channel_y, session_id,
                    time_window=trial_window
                )

                if not trial_lfps_x or not trial_lfps_y:
                    print(f"      {pair_label}: no trial data")
                    continue

                fs = trial_lfps_x[0]['fs']
                n_trials_pair = min(len(trial_lfps_x), len(trial_lfps_y))
                print(f"      {pair_label} ({channel_x}/{channel_y}): "
                      f"{n_trials_pair} trials, fs={fs:.0f} Hz")

                # --- All-trial coherence (for ANOVA stats + all-trial coherograms) ---
                (per_trial_coh, per_trial_icoh,
                 coh_tfr, icoh_tfr, an_times) = \
                    compute_baseline_normalized_coherence(
                        trial_lfps_x, trial_lfps_y, fs, freqs, n_cycles,
                        baseline_window, analysis_window, trial_window,
                        band_masks
                    )

                if not per_trial_coh:
                    continue

                # Cache all-trial coherogram TFR
                if phase_name not in coh_cache[pair_label]:
                    coh_cache[pair_label][phase_name] = {
                        'z_coh_list': [],
                        'z_icoh_list': [],
                        'analysis_times': an_times,
                        'freqs': freqs,
                        'n_trials_list': [],
                    }
                cache_entry = coh_cache[pair_label][phase_name]
                cache_entry['z_coh_list'].append(coh_tfr)
                cache_entry['z_icoh_list'].append(icoh_tfr)
                cache_entry['n_trials_list'].append(n_trials_pair)

                # --- CS-alone (probe) trial coherograms ---
                csalone_x = [t for t in trial_lfps_x
                             if t['trial_type'] == 'cs_alone']
                csalone_y = [t for t in trial_lfps_y
                             if t['trial_type'] == 'cs_alone']
                n_csalone = min(len(csalone_x), len(csalone_y))

                if n_csalone >= 2:
                    (_, _, csa_coh_tfr, csa_icoh_tfr, csa_times) = \
                        compute_baseline_normalized_coherence(
                            csalone_x, csalone_y, fs, freqs, n_cycles,
                            baseline_window, analysis_window, trial_window,
                            band_masks
                        )
                    if csa_coh_tfr.size > 0:
                        if phase_name not in coh_cache_csalone[pair_label]:
                            coh_cache_csalone[pair_label][phase_name] = {
                                'z_coh_list': [],
                                'z_icoh_list': [],
                                'analysis_times': csa_times,
                                'freqs': freqs,
                                'n_trials_list': [],
                            }
                        csa_entry = coh_cache_csalone[pair_label][phase_name]
                        csa_entry['z_coh_list'].append(csa_coh_tfr)
                        csa_entry['z_icoh_list'].append(csa_icoh_tfr)
                        csa_entry['n_trials_list'].append(n_csalone)
                        print(f"        CS-alone: {n_csalone} probe trials")

                # Build long-format rows (all trials for ANOVA)
                for band_name in freq_bands.keys():
                    trial_z_coh = per_trial_coh[band_name]
                    trial_z_icoh = per_trial_icoh[band_name]
                    for trial_idx in range(len(trial_z_coh)):
                        t_num = trial_lfps_x[trial_idx]['trial_number']
                        rows.append({
                            'animal': animal_id,
                            'phase': phase_name,
                            'pair': pair_label,
                            'band': band_name,
                            'trial_number': t_num,
                            'session': session_id,
                            'z_coherence': trial_z_coh[trial_idx],
                            'z_icoh': trial_z_icoh[trial_idx],
                        })

    # Weighted average of coherogram TFRs across sessions within each phase
    for cache in [coh_cache, coh_cache_csalone]:
        for pair_label in cache:
            for phase_name in list(cache[pair_label].keys()):
                entry = cache[pair_label][phase_name]
                if entry.get('z_coh_list'):
                    weights = np.array(entry['n_trials_list'], dtype=float)
                    total = weights.sum()
                    z_coh_avg = sum(w * c for w, c in
                                    zip(weights, entry['z_coh_list'])) / total
                    z_icoh_avg = sum(w * c for w, c in
                                     zip(weights, entry['z_icoh_list'])) / total
                    entry['z_coh_avg'] = z_coh_avg
                    entry['z_icoh_avg'] = z_icoh_avg
                    entry['n_trials'] = int(total)
                # Clean up accumulation lists
                for key in ['z_coh_list', 'z_icoh_list', 'n_trials_list']:
                    if key in entry:
                        del entry[key]

    coh_results_df = pd.DataFrame(rows)
    return coh_results_df, coh_cache, coh_cache_csalone


# ============================================================
# Statistical Analysis (NEW)
# ============================================================

def run_phase_anova(results_df, freq_bands, regions, phase_order, trim_sd=2.0):
    """
    One-way ANOVA across phases for each region x band.
    Iterative outlier trimming, LSD post-hoc, partial eta-squared.

    Returns
    -------
    anova_df : pd.DataFrame
        One row per region x band.
    posthoc_df : pd.DataFrame
        All significant pairwise comparisons.
    """
    anova_rows = []
    posthoc_frames = []

    for region_name in regions.keys():
        for band_name in freq_bands.keys():
            subset = results_df[
                (results_df['region'] == region_name) &
                (results_df['band'] == band_name)
            ].copy()

            if subset.empty or subset['phase'].nunique() < 2:
                continue

            # Iterative outlier trimming (mean +/- 2 SD)
            trimmed = subset.copy()
            prev_len = 0
            while len(trimmed) != prev_len:
                prev_len = len(trimmed)
                mean_val = trimmed['z_power'].mean()
                std_val = trimmed['z_power'].std()
                if std_val == 0:
                    break
                trimmed = trimmed[
                    (trimmed['z_power'] >= mean_val - trim_sd * std_val) &
                    (trimmed['z_power'] <= mean_val + trim_sd * std_val)
                ]

            if trimmed['phase'].nunique() < 2:
                continue

            n_trimmed = len(subset) - len(trimmed)

            # One-way ANOVA
            try:
                aov = pg.anova(
                    data=trimmed,
                    dv='z_power',
                    between='phase',
                    detailed=True
                )

                f_val = aov.loc[0, 'F']
                p_val = aov.loc[0, 'p-unc']
                np2 = aov.loc[0, 'np2']
                df_between = int(aov.loc[0, 'DF'])
                df_within = int(aov.loc[1, 'DF']) if len(aov) > 1 else np.nan

                anova_rows.append({
                    'region': region_name,
                    'band': band_name,
                    'F': f_val,
                    'p': p_val,
                    'np2': np2,
                    'df_between': df_between,
                    'df_within': df_within,
                    'n_trials': len(trimmed),
                    'n_trimmed': n_trimmed,
                    'significant': p_val < 0.05,
                })

                # LSD post-hoc (uncorrected pairwise t-tests)
                if p_val < 0.05:
                    posthoc = pg.pairwise_tests(
                        data=trimmed,
                        dv='z_power',
                        between='phase',
                        parametric=True,
                        padjust='none',
                        effsize='hedges',
                        return_desc=True,
                    )
                    posthoc['region'] = region_name
                    posthoc['band'] = band_name
                    posthoc_frames.append(posthoc)

            except Exception as e:
                print(f"    ANOVA failed for {region_name}/{band_name}: {e}")

    anova_df = pd.DataFrame(anova_rows)
    posthoc_df = (pd.concat(posthoc_frames, ignore_index=True)
                  if posthoc_frames else pd.DataFrame())

    return anova_df, posthoc_df


def run_coherence_anova(coh_results_df, freq_bands, region_pairs, phase_order,
                         trim_sd=2.0):
    """
    One-way ANOVA across phases for each region-pair x band.
    Same methodology as run_phase_anova: iterative outlier trimming,
    LSD post-hoc, partial eta-squared.

    Parameters
    ----------
    coh_results_df : pd.DataFrame
        Must have columns: [pair, band, phase, z_coherence]
    freq_bands : dict
    region_pairs : list of (str, str) tuples
    phase_order : list
    trim_sd : float

    Returns
    -------
    anova_df : pd.DataFrame (one row per pair x band)
    posthoc_df : pd.DataFrame
    """
    anova_rows = []
    posthoc_frames = []

    pair_labels = [f'{r1}-{r2}' for r1, r2 in region_pairs]

    for pair_label in pair_labels:
        for band_name in freq_bands.keys():
            subset = coh_results_df[
                (coh_results_df['pair'] == pair_label) &
                (coh_results_df['band'] == band_name)
            ].copy()

            if subset.empty or subset['phase'].nunique() < 2:
                continue

            # Iterative outlier trimming (mean +/- 2 SD)
            trimmed = subset.copy()
            prev_len = 0
            while len(trimmed) != prev_len:
                prev_len = len(trimmed)
                mean_val = trimmed['z_coherence'].mean()
                std_val = trimmed['z_coherence'].std()
                if std_val == 0:
                    break
                trimmed = trimmed[
                    (trimmed['z_coherence'] >= mean_val - trim_sd * std_val) &
                    (trimmed['z_coherence'] <= mean_val + trim_sd * std_val)
                ]

            if trimmed['phase'].nunique() < 2:
                continue

            n_trimmed = len(subset) - len(trimmed)

            # One-way ANOVA
            try:
                aov = pg.anova(
                    data=trimmed,
                    dv='z_coherence',
                    between='phase',
                    detailed=True
                )

                f_val = aov.loc[0, 'F']
                p_val = aov.loc[0, 'p-unc']
                np2 = aov.loc[0, 'np2']
                df_between = int(aov.loc[0, 'DF'])
                df_within = int(aov.loc[1, 'DF']) if len(aov) > 1 else np.nan

                anova_rows.append({
                    'pair': pair_label,
                    'band': band_name,
                    'F': f_val,
                    'p': p_val,
                    'np2': np2,
                    'df_between': df_between,
                    'df_within': df_within,
                    'n_trials': len(trimmed),
                    'n_trimmed': n_trimmed,
                    'significant': p_val < 0.05,
                })

                # LSD post-hoc (uncorrected pairwise t-tests)
                if p_val < 0.05:
                    posthoc = pg.pairwise_tests(
                        data=trimmed,
                        dv='z_coherence',
                        between='phase',
                        parametric=True,
                        padjust='none',
                        effsize='hedges',
                        return_desc=True,
                    )
                    posthoc['pair'] = pair_label
                    posthoc['band'] = band_name
                    posthoc_frames.append(posthoc)

            except Exception as e:
                print(f"    Coherence ANOVA failed for "
                      f"{pair_label}/{band_name}: {e}")

    anova_df = pd.DataFrame(anova_rows)
    posthoc_df = (pd.concat(posthoc_frames, ignore_index=True)
                  if posthoc_frames else pd.DataFrame())

    return anova_df, posthoc_df


# ============================================================
# Plotting Functions (NEW)
# ============================================================

def plot_phase_comparison(results_df, anova_df, posthoc_df, regions,
                           freq_bands, phase_order, animal_id, figsize=None):
    """
    Bar plots: z-scored power across phases, per region x band.
    Grid layout: rows = bands, cols = regions.
    """
    band_names = list(freq_bands.keys())
    region_names = list(regions.keys())
    n_bands = len(band_names)
    n_regions = len(region_names)

    if figsize is None:
        figsize = (5 * n_regions, 4 * n_bands)

    fig, axes = plt.subplots(n_bands, n_regions, figsize=figsize, squeeze=False)

    phase_colors = {
        'UP': '#808080', 'IT': '#5B9BD5',
        'LRN': '#70AD47', 'RET': '#FFC000',
    }

    for row, band_name in enumerate(band_names):
        for col, region_name in enumerate(region_names):
            ax = axes[row, col]

            subset = results_df[
                (results_df['region'] == region_name) &
                (results_df['band'] == band_name)
            ]

            means, sems = [], []
            for phase in phase_order:
                phase_data = subset[subset['phase'] == phase]['z_power']
                means.append(phase_data.mean() if len(phase_data) > 0 else 0)
                sems.append(phase_data.sem() if len(phase_data) > 1 else 0)

            x = np.arange(len(phase_order))
            colors = [phase_colors.get(p, 'gray') for p in phase_order]

            ax.bar(x, means, yerr=sems, capsize=4,
                   color=colors, edgecolor='black', linewidth=1.2,
                   zorder=3, alpha=0.8)

            # Individual trial points (jittered)
            for i, phase in enumerate(phase_order):
                vals = subset[subset['phase'] == phase]['z_power'].values
                if len(vals) > 0:
                    jitter = np.random.normal(0, 0.06, len(vals))
                    ax.scatter(np.full(len(vals), i) + jitter, vals,
                              color='black', s=6, alpha=0.25, zorder=4)

            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(phase_order, fontsize=10, fontweight='bold')
            ax.set_ylabel('Z-scored Power', fontsize=11)
            ax.set_title(f'{region_name} - {band_name}', fontsize=12,
                        fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

            # Annotate ANOVA result
            aov_row = anova_df[
                (anova_df['region'] == region_name) &
                (anova_df['band'] == band_name)
            ]
            if not aov_row.empty:
                f_val = aov_row.iloc[0]['F']
                p_val = aov_row.iloc[0]['p']
                np2_val = aov_row.iloc[0]['np2']
                sig_str = '*' if p_val < 0.05 else 'ns'
                ax.text(0.02, 0.95,
                       f'F={f_val:.2f}, p={p_val:.4f} {sig_str}\n'
                       f'\u03b7\u00b2p={np2_val:.3f}',
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f'Z-scored Power Across Learning Phases - {animal_id}\n'
                 f'(Baseline-normalized, post-CS)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'phase_comparison_{animal_id}.png')


def _draw_event_markers(ax, phase_name, style='spectrogram'):
    """Draw CS/trace/US event markers. UP phase has no trace or US."""
    if style == 'spectrogram':
        # On colored backgrounds — use white/cyan
        ax.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.9,
                   label='CS onset')
        ax.axvline(250, color='white', linestyle=':', linewidth=1.5, alpha=0.8,
                   label='CS offset')
        if phase_name != 'UP':
            ax.axvline(750, color='cyan', linestyle='--', linewidth=2, alpha=0.9,
                       label='US onset')
            ax.axvline(775, color='cyan', linestyle=':', linewidth=1.2, alpha=0.7,
                       label='US offset')
    else:
        # On white backgrounds (line plots)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='CS onset')
        ax.axvline(250, color='black', linestyle=':', linewidth=1, alpha=0.5,
                   label='CS offset')
        if phase_name != 'UP':
            ax.axvline(750, color='red', linestyle='--', linewidth=1.5, alpha=0.6,
                       label='US onset')
            ax.axvline(775, color='red', linestyle=':', linewidth=1, alpha=0.4,
                       label='US offset')


def plot_morlet_spectrograms(tfr_cache, freqs, regions, phase_order,
                              freq_range=(4, 50), animal_id='', figsize=None):
    """
    Time-frequency spectrograms: rows = regions, cols = phases.
    Uses z-scored Morlet wavelet power.
    """
    region_names = list(regions.keys())
    n_phases = len(phase_order)
    n_regions = len(region_names)

    if figsize is None:
        figsize = (5 * n_phases, 4 * n_regions)

    fig, axes = plt.subplots(n_regions, n_phases, figsize=figsize,
                              squeeze=False, sharex=True, sharey=True)

    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    im = None

    # First pass: collect all data to compute shared color limits
    all_spec_data = []
    for col, phase_name in enumerate(phase_order):
        for row, region_name in enumerate(region_names):
            all_z = []
            if phase_name in tfr_cache and region_name in tfr_cache[phase_name]:
                for channel in tfr_cache[phase_name][region_name]:
                    cache = tfr_cache[phase_name][region_name][channel]
                    if 'z_power_avg' in cache:
                        all_z.append(cache['z_power_avg'])
            if all_z:
                avg_z = np.mean(all_z, axis=0)
                avg_z_smooth = gaussian_filter(avg_z, sigma=2.0)
                all_spec_data.append(avg_z_smooth[freq_mask, :])

    # Compute shared color limits from 5th/95th percentile across all panels
    if all_spec_data:
        all_vals = np.concatenate([s.ravel() for s in all_spec_data])
        vmin = np.percentile(all_vals, 5)
        vmax = np.percentile(all_vals, 95)
    else:
        vmin, vmax = -3, 3

    # Second pass: plot
    for col, phase_name in enumerate(phase_order):
        for row, region_name in enumerate(region_names):
            ax = axes[row, col]

            all_z = []
            analysis_times = None

            if phase_name in tfr_cache and region_name in tfr_cache[phase_name]:
                for channel in tfr_cache[phase_name][region_name]:
                    cache = tfr_cache[phase_name][region_name][channel]
                    if 'z_power_avg' in cache:
                        all_z.append(cache['z_power_avg'])
                        analysis_times = cache['analysis_times']

            if all_z and analysis_times is not None:
                avg_z = np.mean(all_z, axis=0)
                avg_z_smooth = gaussian_filter(avg_z, sigma=2.0)

                t_ms = analysis_times * 1000

                im = ax.pcolormesh(
                    t_ms, freqs[freq_mask], avg_z_smooth[freq_mask, :],
                    shading='gouraud', cmap='RdBu_r',
                    vmin=vmin, vmax=vmax
                )

                # Phase-aware event markers
                _draw_event_markers(ax, phase_name, style='spectrogram')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, style='italic')

            if col == 0:
                ax.set_ylabel(f'{region_name}\nFreq (Hz)',
                            fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(phase_name, fontsize=13, fontweight='bold')
            if row == n_regions - 1:
                ax.set_xlabel('Time from CS (ms)', fontsize=11)

    # Colorbar
    if im is not None:
        plt.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Z-scored Power', fontsize=12, fontweight='bold',
                       rotation=270, labelpad=15)

    freq_label = f'{freq_range[0]}-{freq_range[1]} Hz'
    fig.suptitle(f'Morlet Spectrograms ({freq_label}) - {animal_id}\n'
                 f'(Z-scored vs pre-CS baseline)',
                 fontsize=15, fontweight='bold', y=0.98)
    safe_name = animal_id.replace(' ', '_').replace('(', '').replace(')', '')
    save_figure(fig, f'spectrograms_{freq_range[0]}-{freq_range[1]}Hz_{safe_name}.png')


def plot_band_power_timecourse(tfr_cache, freqs, band_masks, regions,
                                 phase_order, animal_id='', figsize=None):
    """
    Line plots: z-scored band power over time.
    Rows = bands, cols = regions, lines = phases overlaid.
    """
    band_names = list(band_masks.keys())
    region_names = list(regions.keys())
    n_bands = len(band_names)
    n_regions = len(region_names)

    phase_colors = {
        'UP': '#808080', 'IT': '#5B9BD5',
        'LRN': '#70AD47', 'RET': '#FFC000',
    }

    if figsize is None:
        figsize = (6 * n_regions, 4 * n_bands)

    fig, axes = plt.subplots(n_bands, n_regions, figsize=figsize,
                              squeeze=False, sharex=True)

    for row, band_name in enumerate(band_names):
        mask = band_masks[band_name]

        for col, region_name in enumerate(region_names):
            ax = axes[row, col]

            for phase_name in phase_order:
                all_band_power = []
                analysis_times = None

                if (phase_name in tfr_cache and
                    region_name in tfr_cache[phase_name]):
                    for channel in tfr_cache[phase_name][region_name]:
                        cache = tfr_cache[phase_name][region_name][channel]
                        if 'z_power_avg' in cache:
                            # Average over freq bins in this band
                            band_tc = cache['z_power_avg'][mask, :].mean(axis=0)
                            all_band_power.append(band_tc)
                            analysis_times = cache['analysis_times']

                if all_band_power and analysis_times is not None:
                    avg_power = np.mean(all_band_power, axis=0)
                    t_ms = analysis_times * 1000
                    ax.plot(t_ms, avg_power, color=phase_colors[phase_name],
                           linewidth=2, label=phase_name, alpha=0.9)

            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            # Event markers (trace/US shown for paired phases)
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(250, color='black', linestyle=':', linewidth=1, alpha=0.5)
            # Only show trace/US markers if non-UP phases are present
            has_paired = any(p != 'UP' for p in phase_order)
            if has_paired:
                ax.axvline(750, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
                ax.axvline(775, color='red', linestyle=':', linewidth=1, alpha=0.4)

            if row == 0:
                ax.set_title(region_name, fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{band_name}\nZ-scored Power',
                            fontsize=11, fontweight='bold')
            if row == n_bands - 1:
                ax.set_xlabel('Time from CS (ms)', fontsize=11)
            if row == 0 and col == n_regions - 1:
                ax.legend(fontsize=9, framealpha=0.9)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'Band Power Time Course Across Phases - {animal_id}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'band_timecourse_{animal_id}.png')


# ============================================================
# Coherence Plotting Functions (Phase 2)
# ============================================================

def plot_coherence_comparison(coh_results_df, anova_df, posthoc_df,
                                region_pairs, freq_bands, phase_order,
                                animal_id, figsize=None):
    """
    Bar plots: z-scored coherence across phases.
    Grid: rows = frequency bands, cols = region pairs.
    """
    band_names = list(freq_bands.keys())
    pair_labels = [f'{r1}-{r2}' for r1, r2 in region_pairs]
    n_bands = len(band_names)
    n_pairs = len(pair_labels)

    if figsize is None:
        figsize = (5 * n_pairs, 4 * n_bands)

    fig, axes = plt.subplots(n_bands, n_pairs, figsize=figsize, squeeze=False)

    phase_colors = {
        'UP': '#808080', 'IT': '#5B9BD5',
        'LRN': '#70AD47', 'RET': '#FFC000',
    }

    for row, band_name in enumerate(band_names):
        for col, pair_label in enumerate(pair_labels):
            ax = axes[row, col]

            subset = coh_results_df[
                (coh_results_df['pair'] == pair_label) &
                (coh_results_df['band'] == band_name)
            ]

            means, sems = [], []
            for phase in phase_order:
                phase_data = subset[subset['phase'] == phase]['z_coherence']
                means.append(phase_data.mean() if len(phase_data) > 0 else 0)
                sems.append(phase_data.sem() if len(phase_data) > 1 else 0)

            x = np.arange(len(phase_order))
            colors = [phase_colors.get(p, 'gray') for p in phase_order]

            ax.bar(x, means, yerr=sems, capsize=4,
                   color=colors, edgecolor='black', linewidth=1.2,
                   zorder=3, alpha=0.8)

            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(phase_order, fontsize=10, fontweight='bold')
            ax.set_ylabel('Z-scored Coherence', fontsize=11)

            # Tight y-axis based on bars + error bars
            all_tops = [m + s for m, s in zip(means, sems)]
            all_bots = [m - s for m, s in zip(means, sems)]
            y_max = max(all_tops) if all_tops else 1
            y_min = min(all_bots) if all_bots else -1
            y_pad = max(0.3, (y_max - y_min) * 0.25)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

            # Format pair label for display: "ACC ↔ Hippocampus"
            display_pair = pair_label.replace('-', ' \u2194 ')
            ax.set_title(f'{display_pair} - {band_name}', fontsize=12,
                        fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

            # Annotate ANOVA result
            aov_row = anova_df[
                (anova_df['pair'] == pair_label) &
                (anova_df['band'] == band_name)
            ]
            if not aov_row.empty:
                f_val = aov_row.iloc[0]['F']
                p_val = aov_row.iloc[0]['p']
                np2_val = aov_row.iloc[0]['np2']
                sig_str = '*' if p_val < 0.05 else 'ns'
                ax.text(0.02, 0.95,
                       f'F={f_val:.2f}, p={p_val:.4f} {sig_str}\n'
                       f'\u03b7\u00b2p={np2_val:.3f}',
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f'Z-scored Coherence Across Learning Phases - {animal_id}\n'
                 f'(Baseline-normalized, post-CS)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'coherence_comparison_{animal_id}.png')


def plot_coherograms(coh_cache, freqs, region_pairs, phase_order,
                       freq_range=(4, 50), animal_id='', figsize=None):
    """
    Time-frequency coherograms: rows = region pairs, cols = phases.
    """
    pair_labels = [f'{r1}-{r2}' for r1, r2 in region_pairs]
    n_pairs = len(pair_labels)
    n_phases = len(phase_order)

    if figsize is None:
        figsize = (5 * n_phases, 4 * n_pairs)

    fig, axes = plt.subplots(n_pairs, n_phases, figsize=figsize,
                              squeeze=False, sharex=True, sharey=True)

    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    im = None

    # First pass: compute shared color limits
    all_spec_data = []
    for pair_label in pair_labels:
        for phase_name in phase_order:
            if (pair_label in coh_cache and
                phase_name in coh_cache[pair_label] and
                'z_coh_avg' in coh_cache[pair_label][phase_name]):
                data = coh_cache[pair_label][phase_name]['z_coh_avg']
                smoothed = gaussian_filter(data, sigma=3.0)
                all_spec_data.append(smoothed[freq_mask, :])

    if all_spec_data:
        all_vals = np.concatenate([s.ravel() for s in all_spec_data])
        vmin = np.percentile(all_vals, 5)
        vmax = np.percentile(all_vals, 95)
    else:
        vmin, vmax = -3, 3

    # Second pass: plot
    for col, phase_name in enumerate(phase_order):
        for row, pair_label in enumerate(pair_labels):
            ax = axes[row, col]

            if (pair_label in coh_cache and
                phase_name in coh_cache[pair_label] and
                'z_coh_avg' in coh_cache[pair_label][phase_name]):

                entry = coh_cache[pair_label][phase_name]
                z_coh = entry['z_coh_avg']
                z_coh_smooth = gaussian_filter(z_coh, sigma=2.0)
                an_times = entry['analysis_times']
                t_ms = an_times * 1000

                im = ax.pcolormesh(
                    t_ms, freqs[freq_mask], z_coh_smooth[freq_mask, :],
                    shading='gouraud', cmap='RdBu_r',
                    vmin=vmin, vmax=vmax
                )

                # Phase-aware event markers
                _draw_event_markers(ax, phase_name, style='spectrogram')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, style='italic')

            if col == 0:
                display_pair = pair_label.replace('-', '\u2194')
                ax.set_ylabel(f'{display_pair}\nFreq (Hz)',
                            fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(phase_name, fontsize=13, fontweight='bold')
            if row == n_pairs - 1:
                ax.set_xlabel('Time from CS (ms)', fontsize=11)

    # Colorbar
    if im is not None:
        plt.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Z-scored Coherence', fontsize=12, fontweight='bold',
                       rotation=270, labelpad=15)

    freq_label = f'{freq_range[0]}-{freq_range[1]} Hz'
    fig.suptitle(f'Coherograms ({freq_label}) - {animal_id}\n'
                 f'(Z-scored vs pre-CS baseline)',
                 fontsize=15, fontweight='bold', y=0.98)
    safe_name = animal_id.replace(' ', '_').replace('(', '').replace(')', '')
    save_figure(fig, f'coherograms_{freq_range[0]}-{freq_range[1]}Hz_{safe_name}.png')


# ============================================================
# ANIMAL & EXPERIMENT CONFIGURATION
# ============================================================

ANIMAL_CONFIGS = {
    'R6642': {
        'base_path': r'X:\Koushani\ephys\R6642',
        'regions': {
            'Cerebellum': ['CSC3'],
            'Hippocampus': ['CSC15'],
            'ACC': ['CSC7'],
        },
        'phases': {
            'UP':  ['KB_R6642__S01_UP_2k'],
            'IT':  ['KB_R6642__S02_Paired_2k', 'KB_R6642__S03_Paired_2k'],
            'LRN': ['KB_R6642__S08_Paired_2k', 'KB_R6642__S09_Paired_2k'],
            'RET': ['KB_R6642__S15_Paired_2k'],
        },
        'cs_ttl': 2,
        'us_ttl': 4,
        'skip_first': 1,
    },
    # Add future animals here:
    # 'R6643': { ... },
}

# Frequency bands matching Halverson, Kim & Freeman (2023)
FREQ_BANDS = {
    'Theta':      (4, 12),
    'Beta':       (15, 25),
    'Slow Gamma': (25, 50),
    'Fast Gamma': (65, 140),
}

PHASE_ORDER = ['UP', 'IT', 'LRN', 'RET']

# Which animals to analyze
ANIMALS_TO_ANALYZE = ['R6642']

# Time windows (seconds relative to CS onset)
# Pre-CS: 200 ms | CS: 250 ms | Trace: 500 ms | US: 25 ms | Post-US: 500 ms
# Timeline:  -0.2 ... 0 ... 0.25 ... 0.75 ... 0.775 ... 1.275
TRIAL_WINDOW = (-2.0, 2.0)       # Wide window for wavelet edge padding
BASELINE_WINDOW = (-0.2, 0.0)    # Pre-CS baseline for z-normalization
ANALYSIS_WINDOW = (-0.2, 1.275)  # Pre-CS through post-US

# Region pairs for coherence analysis (Phase 2)
REGION_PAIRS = [
    ('ACC', 'Hippocampus'),
    ('Cerebellum', 'ACC'),
    ('Cerebellum', 'Hippocampus'),
]


# ============================================================
# MODULAR RUN FUNCTIONS (with pickle caching)
# ============================================================

def run_power_loading(animal_id, config, freqs, n_cycles, band_masks,
                      use_cache=True):
    """STEP 1: Load data & compute Morlet TFR power (cacheable)."""
    cache_name = 'power_data'
    if use_cache:
        cached = load_cache(cache_name, animal_id)
        if cached is not None:
            return (cached['results_df'], cached['tfr_cache'],
                    cached.get('tfr_cache_csalone', {}))

    print("  Computing from scratch (no cache)...")
    results_df, tfr_cache, tfr_cache_csalone = collect_phase_data(
        animal_id, config, FREQ_BANDS, freqs, n_cycles, band_masks,
        TRIAL_WINDOW, BASELINE_WINDOW, ANALYSIS_WINDOW
    )

    if not results_df.empty:
        save_cache({'results_df': results_df, 'tfr_cache': tfr_cache,
                    'tfr_cache_csalone': tfr_cache_csalone},
                   cache_name, animal_id)

    return results_df, tfr_cache, tfr_cache_csalone


def run_power_stats(animal_id, config, results_df):
    """STEP 2: ANOVA + LSD post-hoc on power data."""
    anova_df, posthoc_df = run_phase_anova(
        results_df, FREQ_BANDS, config['regions'], PHASE_ORDER,
        trim_sd=2.0
    )

    print("\n  ANOVA Results:")
    if not anova_df.empty:
        print(anova_df[['region', 'band', 'F', 'p', 'np2', 'n_trials',
                       'significant']].to_string(index=False))
    else:
        print("  No ANOVA results.")

    if not posthoc_df.empty:
        sig_ph = posthoc_df[posthoc_df['p-unc'] < 0.05]
        if not sig_ph.empty:
            print("\n  Significant Post-hoc (LSD):")
            cols = ['region', 'band', 'A', 'B', 'T', 'p-unc', 'hedges']
            available_cols = [c for c in cols if c in sig_ph.columns]
            print(sig_ph[available_cols].to_string(index=False))
        else:
            print("\n  No significant pairwise differences.")

    return anova_df, posthoc_df


def run_power_plots(animal_id, config, results_df, tfr_cache, freqs,
                    band_masks, anova_df, posthoc_df,
                    tfr_cache_csalone=None):
    """STEPS 3-5: Power bar plots, spectrograms, band time courses."""
    # STEP 3: Phase comparison bar plots
    print("\n  STEP 3: Plotting phase comparisons...")
    plot_phase_comparison(
        results_df, anova_df, posthoc_df,
        config['regions'], FREQ_BANDS, PHASE_ORDER, animal_id
    )

    # STEP 4: Morlet spectrograms
    print("\n  STEP 4: Plotting spectrograms...")
    plot_morlet_spectrograms(
        tfr_cache, freqs, config['regions'], PHASE_ORDER,
        freq_range=(4, 50), animal_id=animal_id
    )
    plot_morlet_spectrograms(
        tfr_cache, freqs, config['regions'], PHASE_ORDER,
        freq_range=(65, 140), animal_id=f'{animal_id} (Fast Gamma)'
    )

    # STEP 5a: Band power time courses (all trials)
    print("\n  STEP 5a: Plotting band power time courses (all trials)...")
    plot_band_power_timecourse(
        tfr_cache, freqs, band_masks, config['regions'],
        PHASE_ORDER, animal_id=animal_id
    )

    # STEP 5b: Band power time courses (CS-alone trials only)
    if tfr_cache_csalone:
        print("\n  STEP 5b: Plotting band power time courses (CS-alone)...")
        plot_band_power_timecourse(
            tfr_cache_csalone, freqs, band_masks, config['regions'],
            PHASE_ORDER, animal_id=f'{animal_id}_CSalone'
        )


def run_coherence_loading(animal_id, config, freqs, n_cycles, band_masks,
                          use_cache=True):
    """STEP 6: Compute LFP-LFP coherence across phases (cacheable)."""
    cache_name = 'coherence_data'
    if use_cache:
        cached = load_cache(cache_name, animal_id)
        if cached is not None:
            return (cached['coh_results_df'], cached['coh_cache'],
                    cached['coh_cache_csalone'])

    print("  Computing from scratch (no cache)...")
    coh_results_df, coh_cache, coh_cache_csalone = collect_coherence_data(
        animal_id, config, FREQ_BANDS, freqs, n_cycles, band_masks,
        TRIAL_WINDOW, BASELINE_WINDOW, ANALYSIS_WINDOW, REGION_PAIRS
    )

    if not coh_results_df.empty:
        save_cache({'coh_results_df': coh_results_df,
                    'coh_cache': coh_cache,
                    'coh_cache_csalone': coh_cache_csalone},
                   cache_name, animal_id)

    return coh_results_df, coh_cache, coh_cache_csalone


def run_coherence_stats(coh_results_df):
    """STEP 7: Coherence ANOVA + LSD post-hoc."""
    coh_anova_df, coh_posthoc_df = run_coherence_anova(
        coh_results_df, FREQ_BANDS, REGION_PAIRS, PHASE_ORDER,
        trim_sd=2.0
    )

    print("\n  Coherence ANOVA Results:")
    if not coh_anova_df.empty:
        print(coh_anova_df[['pair', 'band', 'F', 'p', 'np2',
                            'n_trials', 'significant']].to_string(
                                index=False))
    else:
        print("  No coherence ANOVA results.")

    if not coh_posthoc_df.empty:
        sig_ph = coh_posthoc_df[coh_posthoc_df['p-unc'] < 0.05]
        if not sig_ph.empty:
            print("\n  Significant Coherence Post-hoc (LSD):")
            cols = ['pair', 'band', 'A', 'B', 'T', 'p-unc', 'hedges']
            available_cols = [c for c in cols if c in sig_ph.columns]
            print(sig_ph[available_cols].to_string(index=False))
        else:
            print("\n  No significant coherence pairwise differences.")

    return coh_anova_df, coh_posthoc_df


def run_coherence_plots(animal_id, coh_results_df, coh_cache,
                        coh_cache_csalone, freqs,
                        coh_anova_df, coh_posthoc_df):
    """STEPS 8-10: Coherence bar plots + coherograms."""
    # STEP 8: Coherence bar plots
    print("\n  STEP 8: Plotting coherence phase comparisons...")
    plot_coherence_comparison(
        coh_results_df, coh_anova_df, coh_posthoc_df,
        REGION_PAIRS, FREQ_BANDS, PHASE_ORDER, animal_id
    )

    # STEP 9: Coherograms — all trials
    print("\n  STEP 9: Plotting coherograms (all trials)...")
    plot_coherograms(
        coh_cache, freqs, REGION_PAIRS, PHASE_ORDER,
        freq_range=(4, 50), animal_id=animal_id
    )

    # STEP 10: Coherograms — CS-alone (probe) trials only
    print("\n  STEP 10: Plotting coherograms (CS-alone trials)...")
    plot_coherograms(
        coh_cache_csalone, freqs, REGION_PAIRS, PHASE_ORDER,
        freq_range=(4, 50),
        animal_id=f'{animal_id}_CSalone'
    )


# ============================================================
# MAIN EXECUTION
# ============================================================
# Toggle which steps to run. Set to False to skip a step.
# Heavy computation steps (loading) are cached automatically.

RUN_POWER_LOADING   = True   # STEP 1: Load data + Morlet TFR (cached)
RUN_POWER_STATS     = True   # STEP 2: Power ANOVA
RUN_POWER_PLOTS     = True   # STEPS 3-5: Power figures
RUN_COHERENCE_LOADING = True # STEP 6: Coherence computation (cached)
RUN_COHERENCE_STATS = True   # STEP 7: Coherence ANOVA
RUN_COHERENCE_PLOTS = True   # STEPS 8-10: Coherence figures

USE_CACHE = False  # Set False to force recomputation (ignores cached .pkl files)

if __name__ == '__main__':

    print("=" * 60)
    print("LFP ANALYSIS PIPELINE")
    print("Morlet Wavelet + Baseline Z-normalization")
    print("Matching Halverson, Kim & Freeman (2023, J Neurosci)")
    print("=" * 60)

    # Build shared frequency axis
    freqs, band_masks = build_frequency_axis(FREQ_BANDS, n_bins_per_band=30)
    n_cycles = compute_adaptive_n_cycles(freqs, min_cycles=3, scaling_factor=0.5)

    print(f"\nFrequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz "
          f"({len(freqs)} bins)")
    print(f"Bands: {list(FREQ_BANDS.keys())}")
    print(f"Phases: {PHASE_ORDER}")
    print(f"Trial window: {TRIAL_WINDOW}")
    print(f"Baseline: {BASELINE_WINDOW}")
    print(f"Analysis: {ANALYSIS_WINDOW}")
    print(f"Cache: {'ON' if USE_CACHE else 'OFF (recomputing)'}")

    # Per-animal analysis
    all_animal_results = []

    for animal_id in ANIMALS_TO_ANALYZE:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {animal_id}")
        print(f"{'='*60}")

        config = ANIMAL_CONFIGS[animal_id]

        # --- POWER ANALYSIS ---

        results_df = None
        tfr_cache = None
        tfr_cache_csalone = None
        anova_df = None
        posthoc_df = None

        if RUN_POWER_LOADING:
            print("\nSTEP 1: Load data & compute Morlet TFR...")
            results_df, tfr_cache, tfr_cache_csalone = run_power_loading(
                animal_id, config, freqs, n_cycles, band_masks,
                use_cache=USE_CACHE
            )
            if results_df.empty:
                print(f"  No data for {animal_id}, skipping.")
                continue
            print(f"\n  Total trial-band observations: {len(results_df)}")
            for phase in PHASE_ORDER:
                n = len(results_df[results_df['phase'] == phase]) // len(FREQ_BANDS)
                print(f"    {phase}: ~{n} trials")
            all_animal_results.append(results_df)

        if RUN_POWER_STATS and results_df is not None:
            print("\nSTEP 2: Running ANOVA + LSD post-hoc...")
            anova_df, posthoc_df = run_power_stats(animal_id, config, results_df)

        if RUN_POWER_PLOTS and results_df is not None and tfr_cache is not None:
            print("\nSTEPS 3-5: Generating power figures...")
            if anova_df is None:
                # Need stats for bar plot annotations — run them
                anova_df, posthoc_df = run_power_stats(
                    animal_id, config, results_df)
            run_power_plots(animal_id, config, results_df, tfr_cache, freqs,
                            band_masks, anova_df, posthoc_df,
                            tfr_cache_csalone=tfr_cache_csalone)

        # --- COHERENCE ANALYSIS ---

        coh_results_df = None
        coh_cache = None
        coh_cache_csalone = None
        coh_anova_df = None
        coh_posthoc_df = None

        if RUN_COHERENCE_LOADING:
            print("\nSTEP 6: Computing LFP-LFP coherence across phases...")
            coh_results_df, coh_cache, coh_cache_csalone = \
                run_coherence_loading(
                    animal_id, config, freqs, n_cycles, band_masks,
                    use_cache=USE_CACHE
                )
            if not coh_results_df.empty:
                print(f"\n  Total coherence observations: "
                      f"{len(coh_results_df)}")
                for pair in REGION_PAIRS:
                    pair_label = f'{pair[0]}-{pair[1]}'
                    n = len(coh_results_df[
                        coh_results_df['pair'] == pair_label])
                    print(f"    {pair_label}: {n} observations")

        if (RUN_COHERENCE_STATS and coh_results_df is not None
                and not coh_results_df.empty):
            print("\nSTEP 7: Running coherence ANOVA + LSD post-hoc...")
            coh_anova_df, coh_posthoc_df = run_coherence_stats(coh_results_df)

        if (RUN_COHERENCE_PLOTS and coh_results_df is not None
                and not coh_results_df.empty):
            print("\nSTEPS 8-10: Generating coherence figures...")
            if coh_anova_df is None:
                coh_anova_df, coh_posthoc_df = run_coherence_stats(
                    coh_results_df)
            run_coherence_plots(animal_id, coh_results_df, coh_cache,
                                coh_cache_csalone, freqs,
                                coh_anova_df, coh_posthoc_df)

        if (coh_results_df is None or coh_results_df.empty) \
                and RUN_COHERENCE_LOADING:
            print("  No coherence data computed, skipping coherence plots.")

    # Multi-animal pooling (future)
    if len(all_animal_results) > 1:
        combined_df = pd.concat(all_animal_results, ignore_index=True)
        print(f"\nCombined: {combined_df['animal'].nunique()} animals, "
              f"{len(combined_df)} observations")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
