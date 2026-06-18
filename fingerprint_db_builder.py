"""
Build fingerprint DB using SQLite for memory efficiency.
"""
import numpy as np
import librosa
import sqlite3, os, time, pickle
from pathlib import Path
from scipy.ndimage import maximum_filter
from collections import defaultdict

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
PEAK_NBHD = 20
PEAK_NBHD_T = 10
AMP_MIN = -60
FAN_VALUE = 15
MIN_DT = 1
MAX_DT = 100

SONG_DB_DIR = "/home/claude/song_db/EE200 Project Song Database"
DB_PATH = "/home/claude/fingerprints.db"

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

def build_sqlite_db(song_dir=SONG_DB_DIR, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS songs")
    c.execute("DROP TABLE IF EXISTS fingerprints")
    c.execute("""CREATE TABLE songs 
                 (id INTEGER PRIMARY KEY, name TEXT)""")
    c.execute("""CREATE TABLE fingerprints 
                 (hash_f1 INTEGER, hash_f2 INTEGER, hash_dt INTEGER,
                  song_id INTEGER, offset INTEGER)""")
    conn.commit()

    mp3_files = sorted(Path(song_dir).glob("*.mp3"))
    print(f"Found {len(mp3_files)} songs.")
    
    song_id_map = {}
    for idx, mp3 in enumerate(mp3_files):
        song_name = mp3.stem
        c.execute("INSERT INTO songs (id, name) VALUES (?, ?)", (idx, song_name))
        song_id_map[song_name] = idx
    conn.commit()

    for idx, mp3 in enumerate(mp3_files):
        song_name = mp3.stem
        song_id = song_id_map[song_name]
        print(f"  [{idx+1:02d}/50] Indexing: {song_name} …", end=" ", flush=True)
        t0 = time.time()
        y = load_audio(str(mp3))
        S = compute_spectrogram(y)
        fi, ti = get_constellation(S)
        hashes = generate_hashes(fi, ti)
        rows = [(h[0], h[1], h[2], song_id, off) for h, off in hashes]
        c.executemany("INSERT INTO fingerprints VALUES (?,?,?,?,?)", rows)
        conn.commit()
        print(f"{len(hashes):,} hashes ({time.time()-t0:.1f}s)")

    print("Creating index on fingerprints…")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints (hash_f1, hash_f2, hash_dt)")
    conn.commit()
    conn.close()
    print("Database built and indexed.")
    
    # Save song list separately for quick access
    songs = [mp3.stem for mp3 in sorted(Path(song_dir).glob("*.mp3"))]
    with open('/home/claude/song_list.pkl', 'wb') as f:
        pickle.dump(songs, f)
    return song_id_map

if __name__ == "__main__":
    t0 = time.time()
    song_map = build_sqlite_db()
    print(f"Total time: {time.time()-t0:.1f}s")
