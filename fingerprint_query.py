"""
Query engine for Sonic Signatures fingerprinting system.
Works with SQLite database built by fingerprint_db_builder.py
"""
import numpy as np
import librosa
import sqlite3
import pickle
import time
from collections import defaultdict
from scipy.ndimage import maximum_filter
from pathlib import Path

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
PEAK_NBHD = 20
PEAK_NBHD_T = 10
AMP_MIN = -60
FAN_VALUE = 15
MIN_DT = 1
MAX_DT = 100

DB_PATH = "fingerprints.db"
SONG_LIST_PATH = "song_list.pkl"
SONG_DB_DIR = "/home/claude/song_db/EE200 Project Song Database"

def load_audio(path, sr=SR, offset=0.0, duration=None):
    y, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return y

def compute_spectrogram(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def get_constellation(S_db, nbhd_freq=PEAK_NBHD, nbhd_time=PEAK_NBHD_T, amp_min=AMP_MIN):
    struct = np.ones((nbhd_freq, nbhd_time), dtype=bool)
    local_max = maximum_filter(S_db, footprint=struct) == S_db
    detected = local_max & (S_db > amp_min)
    freq_idx, time_idx = np.where(detected)
    order = np.argsort(time_idx)
    return freq_idx[order], time_idx[order]

def generate_hashes(freq_idx, time_idx, fan_value=FAN_VALUE,
                    min_dt=MIN_DT, max_dt=MAX_DT):
    hashes = []
    n_peaks = len(freq_idx)
    for i in range(n_peaks):
        for j in range(i + 1, min(i + fan_value + 1, n_peaks)):
            dt = int(time_idx[j]) - int(time_idx[i])
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            h = (int(freq_idx[i]), int(freq_idx[j]), dt)
            hashes.append((h, int(time_idx[i])))
    return hashes

def match_query_sqlite(query_audio, db_path=DB_PATH, top_n=5):
    """Fingerprint query clip and match against SQLite DB."""
    S = compute_spectrogram(query_audio)
    fi, ti = get_constellation(S)
    hashes = generate_hashes(fi, ti)
    
    if not hashes:
        return []
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Load song id → name map
    c.execute("SELECT id, name FROM songs")
    id_to_name = {row[0]: row[1] for row in c.fetchall()}
    
    # Vote accumulator: song_id → offset_delta → count
    candidates = defaultdict(lambda: defaultdict(int))
    
    # Batch lookup
    for h, q_offset in hashes:
        c.execute(
            "SELECT song_id, offset FROM fingerprints WHERE hash_f1=? AND hash_f2=? AND hash_dt=?",
            h
        )
        rows = c.fetchall()
        for song_id, db_offset in rows:
            delta = db_offset - q_offset
            candidates[song_id][delta] += 1
    
    conn.close()
    
    scores = {}
    for song_id, offset_hist in candidates.items():
        scores[id_to_name[song_id]] = max(offset_hist.values())
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

def get_offset_histogram_sqlite(query_audio, song_name, db_path=DB_PATH):
    """Get offset histogram for a specific song."""
    S = compute_spectrogram(query_audio)
    fi, ti = get_constellation(S)
    hashes = generate_hashes(fi, ti)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id FROM songs WHERE name=?", (song_name,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return {}
    song_id = row[0]
    
    offset_hist = defaultdict(int)
    for h, q_offset in hashes:
        c.execute(
            "SELECT offset FROM fingerprints WHERE hash_f1=? AND hash_f2=? AND hash_dt=? AND song_id=?",
            (h[0], h[1], h[2], song_id)
        )
        for (db_offset,) in c.fetchall():
            offset_hist[db_offset - q_offset] += 1
    
    conn.close()
    return dict(offset_hist)

print("Query engine loaded.")
