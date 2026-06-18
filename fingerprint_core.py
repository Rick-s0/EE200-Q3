"""
Sonic Signatures — Core fingerprinting engine
Implements spectrogram → constellation → hash → match pipeline
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure
from scipy.ndimage import label
import os, pickle, hashlib, time
from collections import defaultdict
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
SR          = 22050          # sample rate
N_FFT       = 2048           # FFT window size
HOP_LENGTH  = 512            # hop between windows
N_MELS      = 128            # mel bins (not used for fingerprint, but for display)
PEAK_NBHD   = 20             # local max neighbourhood (frequency bins)
PEAK_NBHD_T = 10             # neighbourhood in time
AMP_MIN     = -60            # dB threshold for peak acceptance
FAN_VALUE   = 15             # how many pairs per anchor peak
MIN_DT      = 1              # min time gap between anchor and target (frames)
MAX_DT      = 100            # max time gap

SONG_DB_DIR = "/home/claude/song_db/EE200 Project Song Database"

# ── Spectrogram ────────────────────────────────────────────────────────────

def compute_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Short-time Fourier transform magnitude spectrogram, in dB."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def load_audio(path, sr=SR, offset=0.0, duration=None):
    y, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return y

# ── Constellation (peak detection) ─────────────────────────────────────────

def get_constellation(S_db, nbhd_freq=PEAK_NBHD, nbhd_time=PEAK_NBHD_T, amp_min=AMP_MIN):
    """
    Find local maxima in the spectrogram that stand above amp_min.
    Returns (freq_bins, time_frames) arrays.
    """
    struct = np.ones((nbhd_freq, nbhd_time), dtype=bool)
    # local max: each point equals the maximum in its neighbourhood
    local_max = maximum_filter(S_db, footprint=struct) == S_db
    # remove weak peaks
    detected = local_max & (S_db > amp_min)
    freq_idx, time_idx = np.where(detected)
    # sort by time
    order = np.argsort(time_idx)
    return freq_idx[order], time_idx[order]

# ── Hash generation ─────────────────────────────────────────────────────────

def generate_hashes(freq_idx, time_idx, fan_value=FAN_VALUE,
                    min_dt=MIN_DT, max_dt=MAX_DT):
    """
    Pair each anchor peak with its nearest neighbours → (hash, offset) tuples.
    Hash encodes (f_anchor, f_target, delta_t).
    """
    hashes = []
    n_peaks = len(freq_idx)
    for i in range(n_peaks):
        for j in range(i + 1, min(i + fan_value + 1, n_peaks)):
            dt = time_idx[j] - time_idx[i]
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            f1, f2 = freq_idx[i], freq_idx[j]
            h = (int(f1), int(f2), int(dt))
            hashes.append((h, int(time_idx[i])))
    return hashes

# ── Database build ──────────────────────────────────────────────────────────

def build_database(song_dir=SONG_DB_DIR):
    """
    Fingerprint all songs. Returns:
        db  : { hash_tuple : [(song_name, offset_frame), …] }
        song_names : list[str]
    """
    db = defaultdict(list)
    song_names = []
    mp3_files = sorted(Path(song_dir).glob("*.mp3"))
    print(f"Found {len(mp3_files)} songs.")
    for mp3 in mp3_files:
        song_name = mp3.stem
        song_names.append(song_name)
        print(f"  Indexing: {song_name} …", end=" ", flush=True)
        t0 = time.time()
        y = load_audio(str(mp3))
        S = compute_spectrogram(y)
        fi, ti = get_constellation(S)
        hashes = generate_hashes(fi, ti)
        for h, offset in hashes:
            db[h].append((song_name, offset))
        print(f"{len(hashes):,} hashes  ({time.time()-t0:.1f}s)")
    return db, song_names

# ── Query / Match ───────────────────────────────────────────────────────────

def match_query(query_audio, db, top_n=5):
    """
    Fingerprint a query clip, look up hashes in db, vote by offset histogram.
    Returns list of (song_name, score) sorted descending.
    """
    S = compute_spectrogram(query_audio)
    fi, ti = get_constellation(S)
    hashes = generate_hashes(fi, ti)

    # offset histogram per song
    candidates = defaultdict(lambda: defaultdict(int))
    for h, q_offset in hashes:
        if h in db:
            for song_name, db_offset in db[h]:
                delta = db_offset - q_offset
                candidates[song_name][delta] += 1

    # score = max bin count in offset histogram
    scores = {}
    for song_name, offset_hist in candidates.items():
        scores[song_name] = max(offset_hist.values())

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# ── Offset histogram (for visualisation) ───────────────────────────────────

def get_offset_histogram(query_audio, db, song_name):
    """Return offset histogram dict for a specific song (for plotting)."""
    S = compute_spectrogram(query_audio)
    fi, ti = get_constellation(S)
    hashes = generate_hashes(fi, ti)
    offset_hist = defaultdict(int)
    for h, q_offset in hashes:
        if h in db:
            for sn, db_offset in db[h]:
                if sn == song_name:
                    offset_hist[db_offset - q_offset] += 1
    return offset_hist

print("Core fingerprint module loaded.")
