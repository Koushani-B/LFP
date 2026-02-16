# -*- coding: utf-8 -*-
"""
Diagnostic script to identify TTL values in your event files
"""

import numpy as np
import pandas as pd
import os

def read_neuralynx_events(filepath):
    """Read event file (.nev)"""
    dt = np.dtype([('stx', np.int16), ('pkt_id', np.int16), ('pkt_data_size', np.int16),
                   ('timestamp', np.uint64), ('event_id', np.int16), ('ttl', np.int16),
                   ('crc', np.int16), ('dummy1', np.int16), ('dummy2', np.int16),
                   ('extra', np.int32, 8), ('event_string', 'S128')])

    with open(filepath, 'rb') as f:
        f.read(16384)  # Skip header
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
            print(f"Loaded: {nev_file} - {len(df)} events")
        except Exception as e:
            print(f"Error reading {nev_file}: {e}")

    return pd.concat(all_events, ignore_index=True).sort_values('timestamp').reset_index(drop=True) if all_events else pd.DataFrame()

# UPDATE THIS PATH TO YOUR DATA
base_path = r'X:\Koushani\ephys\R6642'
session = 'KB_R6642__S02_Paired_2k'

session_path = os.path.join(base_path, session)
print(f"Reading events from: {session_path}\n")

events_df = read_all_events(session_path)

if events_df.empty:
    print("ERROR: No events found!")
else:
    print(f"\nTotal events: {len(events_df)}")
    print("\n" + "="*60)
    print("TTL VALUE ANALYSIS")
    print("="*60)
    
    # Count each TTL value
    ttl_counts = events_df['ttl'].value_counts().sort_index()
    print("\nTTL Value | Count | Percentage")
    print("-" * 40)
    for ttl, count in ttl_counts.items():
        pct = (count / len(events_df)) * 100
        print(f"TTL {ttl:5d} | {count:5d} | {pct:5.1f}%")
    
    print("\n" + "="*60)
    print("TIMING ANALYSIS (first 20 events)")
    print("="*60)
    print("\nEvent# | Time(s) | TTL | Time since prev (ms)")
    print("-" * 50)
    
    for i in range(min(20, len(events_df))):
        if i == 0:
            print(f"{i+1:6d} | {events_df.loc[i, 'time_seconds']:7.3f} | {events_df.loc[i, 'ttl']:3d} | [start]")
        else:
            time_diff = (events_df.loc[i, 'time_seconds'] - events_df.loc[i-1, 'time_seconds']) * 1000
            print(f"{i+1:6d} | {events_df.loc[i, 'time_seconds']:7.3f} | {events_df.loc[i, 'ttl']:3d} | {time_diff:8.1f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION HINTS")
    print("="*60)
    print("""
Based on trace conditioning paradigm:
- CS (tone) typically lasts 250ms
- Trace interval: 500ms after CS
- US (airpuff) typically lasts 25ms
- Every 10th trial should be CS-alone

Look for:
1. Most common TTL = likely CS (you should have ~90% paired trials)
2. Less common TTL = likely US (appears in ~90% of trials)
3. Inter-event intervals around 750ms suggest CS-US pairing
""")
