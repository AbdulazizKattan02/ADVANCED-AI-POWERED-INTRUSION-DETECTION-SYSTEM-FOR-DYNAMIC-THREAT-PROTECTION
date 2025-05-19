import threading
import time
import random
from scapy.all import sniff
from collections import deque

# === Constants ===
MAX_PACKETS_DISPLAY = 50

# === Shared state ===
captured_packets_data = deque(maxlen=MAX_PACKETS_DISPLAY)
capture_stop_event = threading.Event()

# === Feature names expected by the model ===
FEATURE_NAMES = [
    "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts",
    "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_mean",
    "bwd_pkt_len_mean", "flow_byts_s", "flow_pkts_s",
    "init_win_byts_fwd", "init_win_byts_bwd"
]

def extract_features(packet):
    """
    Generate dummy features from a packet object for live testing.
    """
    try:
        timestamp = time.time()
        pkt_len = len(packet)
        proto = packet.proto if hasattr(packet, "proto") else 0
        src = packet[0].src if hasattr(packet[0], "src") else ""
        dst = packet[0].dst if hasattr(packet[0], "dst") else ""

        features = {
            "flow_duration": round(random.uniform(0.1, 5.0), 3),
            "tot_fwd_pkts": random.randint(1, 10),
            "tot_bwd_pkts": random.randint(0, 10),
            "totlen_fwd_pkts": random.randint(100, 1500),
            "totlen_bwd_pkts": random.randint(50, 1000),
            "fwd_pkt_len_mean": round(random.uniform(50, 500), 2),
            "bwd_pkt_len_mean": round(random.uniform(20, 300), 2),
            "flow_byts_s": round(pkt_len / random.uniform(0.1, 1.5), 2),
            "flow_pkts_s": round((random.randint(1, 5)) / random.uniform(0.1, 1.5), 2),
            "init_win_byts_fwd": random.randint(1000, 8192),
            "init_win_byts_bwd": random.randint(1000, 8192)
        }

        return {
            "timestamp": timestamp,
            "source": src,
            "destination": dst,
            "protocol": proto,
            "length": pkt_len,
            "features": features
        }

    except Exception as e:
        print(f"[extract_features] Error: {e}")
        return None

def run_capture(interface_name="eth0", packet_limit=None):
    def packet_handler(packet):
        if capture_stop_event.is_set():
            return False
        extracted = extract_features(packet)
        if extracted:
            captured_packets_data.append(extracted)

    try:
        print(f"[INFO] Starting sniff on interface: {interface_name}")
        sniff(iface=interface_name, prn=packet_handler, store=False)
    except Exception as e:
        print(f"[ERROR] Failed to sniff on interface {interface_name}: {e}")
