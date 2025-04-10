# flow_predictor.py
import time
from datetime import datetime
from collections import defaultdict, deque
import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
import warnings

warnings.filterwarnings("ignore")

# Load model and preprocessing components
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Expected number of features: 13
# Features: [flow_duration, packet_count, byte_count, avg_pkt_size, std_pkt_size,
#            mean_iat, std_iat, syn_count, ack_count, avg_ttl, avg_win, src_port, dst_port]
EXPECTED_FEATURES = 13

# Dictionary to hold flow statistics per unique flow
flows = defaultdict(lambda: {
    'start_time': None,
    'last_time': None,
    'packet_count': 0,
    'byte_count': 0,
    'packet_sizes': [],
    'iat_list': [],
    'syn_count': 0,
    'ack_count': 0,
    'ttl_list': [],
    'win_list': [],
    'src_port': 0,
    'dst_port': 0,
    'protocol': ""
})

def process_packet(pkt, flows):
    if not pkt.haslayer(IP):
        return

    ip = pkt[IP]
    if pkt.haslayer(TCP):
        proto = "TCP"
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        proto = "UDP"
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        return

    # Create a flow key based on src, dst, ports and protocol
    flow_key = (ip.src, ip.dst, sport, dport, proto)
    now = time.time()
    f = flows[flow_key]

    # Update start and last times, and compute inter-arrival time
    if f['start_time'] is None:
        f['start_time'] = now
    else:
        f['iat_list'].append(now - f['last_time'])
    f['last_time'] = now

    # Accumulate packet stats
    pkt_len = len(pkt)
    f['packet_count'] += 1
    f['byte_count'] += pkt_len
    f['packet_sizes'].append(pkt_len)
    f['ttl_list'].append(ip.ttl)

    if proto == "TCP":
        tcp = pkt[TCP]
        f['win_list'].append(tcp.window)
        flags = tcp.flags
        if flags & 0x02:
            f['syn_count'] += 1
        if flags & 0x10:
            f['ack_count'] += 1

    # Save port and protocol info for prediction
    f['src_port'] = sport
    f['dst_port'] = dport
    f['protocol'] = proto

def extract_flow_features(flows):
    records = []
    for (src, dst, sp, dp, proto), f in flows.items():
        duration = (f['last_time'] - f['start_time']) if f['start_time'] else 0.0
        iats = f['iat_list'] or [0.0]
        pkt_sizes = f['packet_sizes'] or [0.0]
        ttls = f['ttl_list'] or [0.0]
        wins = f['win_list'] or [0.0]

        features = [
            duration,
            f['packet_count'],
            f['byte_count'],
            np.mean(pkt_sizes),
            np.std(pkt_sizes) if len(pkt_sizes) > 1 else 0.0,
            np.mean(iats),
            np.std(iats) if len(iats) > 1 else 0.0,
            f['syn_count'],
            f['ack_count'],
            np.mean(ttls),
            np.mean(wins)
        ]
        # Append src_port and dst_port to reach 13 features
        features.append(sp)
        features.append(dp)
        
        # Ensure exactly 13 features (in case something is missing)
        while len(features) < EXPECTED_FEATURES:
            features.append(0.0)
        records.append((src, dst, proto, features))
    return records

def predict_flows(flow_features):
    for src, dst, proto, feat in flow_features:
        try:
            X = np.array(feat).reshape(1, -1)
            # Replace NaNs with zeros
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)
            pred = model.predict(X_pca)[0]

            if pred != "BENIGN":
                print(f"[⚠️] Attack detected: {pred} from {src} to {dst} ({proto})")
            else:
                print(f"[✅] Flow from {src} to {dst} is benign ({proto})")
        except Exception as e:
            print("[Prediction Error]", e)

def continuous_capture_loop(capture_interval=30):
    print("[🚀] Starting continuous flow-based prediction...")
    try:
        while True:
            # Reset flows for this batch
            flows = defaultdict(lambda: {
                'start_time': None,
                'last_time': None,
                'packet_count': 0,
                'byte_count': 0,
                'packet_sizes': [],
                'iat_list': [],
                'syn_count': 0,
                'ack_count': 0,
                'ttl_list': [],
                'win_list': [],
                'src_port': 0,
                'dst_port': 0,
                'protocol': ""
            })
            print(f"[📡] Capturing for {capture_interval} seconds...")
            sniff(timeout=capture_interval, prn=lambda pkt: process_packet(pkt, flows), store=False)
            print("[🔍] Extracting & predicting flow features...")
            flow_feats = extract_flow_features(flows)
            predict_flows(flow_feats)
            print("[🔁] Restarting capture... Press Ctrl+C to stop\n")
    except KeyboardInterrupt:
        print("\n[🛑] Stopped by user. Exiting...")

if __name__ == "__main__":
    continuous_capture_loop()

