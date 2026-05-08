import threading
import time
import logging
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .library import load_model_checkpoint
from .predict import load_library_tensors

try:
    from scapy.all import sniff
except ImportError:
    raise ImportError("Scapy is required. Please install it using 'pip install scapy'")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    datefmt='%H:%M:%S'
)

DEFAULT_SILENCE_THRESHOLD_BYTES = 20_000
CONFIDENCE_DELTA = 25.0
GENERIC_STREAMING_LABEL = "📺 Generic Streaming (Ambiguous Signal)"
PACKET_COUNT_WEIGHT_REFERENCE = 100


class Phase2LivePipeline:
    """
    Phase-2 Pipeline for VPN Identification: Live Packet Capture and FlowPic Inference.
    
    Features:
    - Multi-threaded: High-speed sniffing + separate processing thread.
    - True sliding window using high-precision time (time_ns).
    - Map IAT and Size to 2D Matrix (32x32 or 64x64).
    - Pixel Normalization 0-255 weighted by total packet count.
    - Live Inference Pipeline (CNN -> 128-d embedding -> Euclidean Distance against Library).
    - Graceful handling of silent periods.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        library_path: str | Path,
        interface: str = None,
        bpf_filter: str = "ip or ip6", # Default to all traffic; common VPN ports: 'udp port 51820' (WireGuard), 'tcp port 443' (OpenVPN)
        window_size: float = 5.0,
        stride: float = 2.0,
        silence_threshold_bytes: int = DEFAULT_SILENCE_THRESHOLD_BYTES,
        max_mtu: int = 1600,
        matrix_size: int = 224, # Must match Phase 1 training (224x224)
        threshold: float = 0.35,
        confidence_delta: float = CONFIDENCE_DELTA,
        packet_count_weight_reference: int = PACKET_COUNT_WEIGHT_REFERENCE,
        gui_callback=None,
        debug_verbose: bool = True
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.window_size = window_size
        self.stride = stride
        self.silence_threshold_bytes = silence_threshold_bytes
        
        # Dynamic Threshold for Gmeet
        if self.bpf_filter and ("19302-19309" in self.bpf_filter or "10000-20000" in self.bpf_filter):
            self.silence_threshold_bytes = 5000
            
        self.max_mtu = max_mtu
        self.bins = matrix_size
        self.threshold = threshold
        self.confidence_delta = confidence_delta
        self.packet_count_weight_reference = max(1, packet_count_weight_reference)
        self.gui_callback = gui_callback
        self.debug_verbose = debug_verbose
        
        # Circular buffer for real-time packet storage
        self.packet_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        self.live_packet_count = 0
        self.running = False
        self.t_sniff = None
        self.t_process = None
        self.t_heartbeat = None

        logging.info("Loading pre-trained model and reference library...")
        try:
            self.model, self.device = load_model_checkpoint(checkpoint_path)
            self.template_library = load_library_tensors(library_path)
            if not self.template_library:
                logging.warning("Reference library is empty! Predictions will all be Unknown.")
        except Exception as e:
            logging.error(f"Failed to load model or library: {e}")
            raise e

    def packet_handler(self, packet):
        """
        Callback for high-speed packet capture.
        Stores exact arrival time (ns), packet length, and source IP.
        """
        arrival_time_ns = time.time_ns()
        length = len(packet)
        src_ip = "Unknown"
        if packet.haslayer('IP'):
            src_ip = packet['IP'].src
        elif packet.haslayer('IPv6'):
            src_ip = packet['IPv6'].src
        
        with self.buffer_lock:
            self.packet_buffer.append((arrival_time_ns, length, src_ip))
            self.live_packet_count += 1

    def sniff_worker(self):
        """
        Thread for sniffing packets in real-time.
        """
        iface_str = self.interface if self.interface else "Default/All"
        logging.info(f"Starting live capture on interface: {iface_str} | BPF: {self.bpf_filter}")
        
        kwargs = {
            "prn": self.packet_handler,
            "store": False,
            "stop_filter": lambda x: not self.running
        }
        if self.interface:
            kwargs["iface"] = self.interface
        if self.bpf_filter:
            kwargs["filter"] = self.bpf_filter
        try:
            sniff(**kwargs)
            logging.info("Live capture thread stopped gracefully.")
        except Exception as e:
            error_msg = f"[!] CRITICAL ERROR in Sniffer: {e}"
            logging.error(error_msg)
            self._report_gui("log", error_msg)

    def process_worker(self):
        """
        Thread for sliding window processing (e.g. 60s window, 15s stride).
        """
        logging.info(f"Starting inference thread (Window: {self.window_size}s, Stride: {self.stride}s)")
        
        while self.running:
            time.sleep(self.stride)
            
            current_time_ns = time.time_ns()
            window_start_ns = current_time_ns - int(self.window_size * 1e9)
            
            with self.buffer_lock:
                # Discard packets older than the sliding window
                while self.packet_buffer and self.packet_buffer[0][0] < window_start_ns:
                    self.packet_buffer.popleft()
                
                # Copy current active window
                current_packets = list(self.packet_buffer)
            
            if not current_packets:
                continue
                
            total_bytes = sum(p[1] for p in current_packets)
            
            if self.debug_verbose:
                debug_msg = f"[DEBUG] Window Packets: {len(current_packets)}. First 5 captured:\n"
                for i, p in enumerate(current_packets[:5]):
                    debug_msg += f"  {i+1}: Src IP: {p[2]}, Size: {p[1]}B\n"
                print(debug_msg)
            
            # Transparent Threshold Logic (Inference Fix)
            if total_bytes < self.silence_threshold_bytes:
                msg = (
                    f"[i] Filtering Noise: {total_bytes}/{self.silence_threshold_bytes} bytes. "
                    "Signal too weak for reliable classification."
                )
                logging.debug(msg)
                self._report_gui("log", msg)
                continue
            
            self.generate_and_predict(current_packets, total_bytes)

    def heartbeat_worker(self):
        """
        Heartbeat & Packet Counter (Visibility Fix).
        """
        last_count = 0
        zero_count_ticks = 0
        
        while self.running:
            time.sleep(2)
            current_count = self.live_packet_count
            
            self._report_gui("log", f"[*] Heartbeat: {current_count} packets captured so far...")
            
            if current_count == last_count:
                zero_count_ticks += 2
                if zero_count_ticks >= 10:
                    if self.bpf_filter and "ip" in self.bpf_filter.lower():
                        alert_msg = "[!] Alert: 0 packets detected for 10+s on 'ip/ip6'. You likely selected the WRONG INTERFACE (e.g., disconnected adapter). Please select your active Wi-Fi/Ethernet interface."
                    else:
                        alert_msg = "[!] Alert: No traffic detected for 10+ seconds. Check BPF Filter/Permissions.\nHint: Try a broader filter like 'ip or ip6' or 'udp'."
                    logging.warning(alert_msg)
                    self._report_gui("log", alert_msg)
            else:
                zero_count_ticks = 0
                
            last_count = current_count

    def generate_and_predict(self, packets, total_bytes):
        """
        Dynamic FlowPic Construction perfectly matched to Phase 1 Training.
        """
        times_ns = np.array([p[0] for p in packets], dtype=np.float64)
        sizes = np.array([p[1] for p in packets], dtype=np.float32)
        
        # 1. Map Time: arrival_time mapped 0.0 to 1.0 (Phase 1 Logic)
        duration = max((times_ns[-1] - times_ns[0]) / 1e9, 1e-9)
        time_values = ((times_ns - times_ns[0]) / 1e9) / duration
            
        sizes = np.clip(sizes, 0, self.max_mtu)
        
        # 2. Histogram 224x224
        matrix, _, _ = np.histogram2d(
            time_values, sizes, 
            bins=[self.bins, self.bins],
            range=[[0.0, 1.0], [0.0, self.max_mtu]]
        )
        matrix = matrix.astype(np.float32)
        
        # 3. Log-Contrast & Percentile Normalization with packet-count weighting.
        matrix = np.log1p(matrix)
        nonzero = matrix[matrix > 0]
        if nonzero.size > 0:
            scale = float(np.quantile(nonzero, 0.995))
            if scale <= 0:
                scale = float(nonzero.max(initial=0.0))
            if scale > 0:
                matrix = np.clip(matrix / scale, 0.0, 1.0)

        packet_count_weight = min(len(packets) / self.packet_count_weight_reference, 1.0)
        matrix = np.clip(matrix * packet_count_weight * 255.0, 0.0, 255.0)
        matrix = matrix.T # Shape: (224, 224)
        
        # Matrix "Sanity Check"
        if matrix.max() == 0:
            error_msg = "[!] Error: Generated FlowPic is empty despite packet count."
            logging.error(error_msg)
            self._report_gui("log", error_msg)
            return
            
        # 4. Live Inference Pipeline
        # Convert to PyTorch Tensor: shape [1, 1, 224, 224]
        tensor = torch.tensor(matrix / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Pass through CNN to extract 128-d embedding
            embedding = self.model(tensor)
            
        # 5. Compare using Euclidean as the primary magnitude check, while
        # retaining cosine distance for angular similarity diagnostics.
        if not self.template_library:
            self._report_gui("inference", "Unknown (No Library)", float('inf'), len(packets), 0.0)
            return

        label, min_dist, confidence = self._compare_live_embedding(embedding, len(packets))
        
        # 6. Output Application Name and Confidence Score
        self._report_gui("inference", label, min_dist, len(packets), confidence)

    def _compare_live_embedding(self, embedding: torch.Tensor, packet_count: int) -> tuple[str, float, float]:
        """
        Rank templates with Euclidean distance and guard close matches with a
        relative confidence delta. Cosine is still computed for dual-distance
        visibility and to preserve the existing unknown threshold semantics.
        """
        unknown = embedding.view(-1).detach()
        unknown_norm = F.normalize(unknown, p=2, dim=0)
        scores: list[dict[str, float | str]] = []

        for label, template_embedding in self.template_library.items():
            template = template_embedding.to(self.device).view(-1)
            template_norm = F.normalize(template, p=2, dim=0)
            euclidean_distance = float(torch.norm(unknown - template, p=2).item())
            
            # Contextual Weighting for Gmeet
            if self.bpf_filter and ("19302-19309" in self.bpf_filter or "10000-20000" in self.bpf_filter):
                if "gmeet" in label.lower():
                    euclidean_distance *= 0.85
                    
            cosine_distance = float(
                1.0 - F.cosine_similarity(
                    unknown_norm.unsqueeze(0),
                    template_norm.unsqueeze(0),
                ).item()
            )
            scores.append(
                {
                    "label": label,
                    "euclidean": euclidean_distance,
                    "cosine": cosine_distance,
                }
            )

        scores.sort(key=lambda item: float(item["euclidean"]))
        best = scores[0]
        second = scores[1] if len(scores) > 1 else None

        # Tie-Breaker Logic (15% Euclidean difference)
        if second:
            d1_euc = float(best["euclidean"])
            d2_euc = float(second["euclidean"])
            denom_euc = max(d2_euc, 1e-12)
            euc_diff_pct = (d2_euc - d1_euc) / denom_euc
            
            if euc_diff_pct < 0.15:
                # Use Cosine Similarity as decision maker (lower is better distance)
                if float(second["cosine"]) < float(best["cosine"]):
                    # Swap them
                    best, second = second, best

        best_label = str(best["label"])
        best_distance = float(best["euclidean"])
        best_cosine = float(best["cosine"])

        confidence = 100.0
        if second:
            d1_euc = float(best["euclidean"])
            d2_euc = float(second["euclidean"])
            denom = max(d2_euc, 1e-12)
            delta = abs(d2_euc - d1_euc) / denom
            confidence = delta * 100.0
            
            if delta < (self.confidence_delta / 100.0):
                best_label = GENERIC_STREAMING_LABEL

        if best_label != GENERIC_STREAMING_LABEL and best_cosine > self.threshold:
            best_label = "Unknown"

        # Confidence Delta Guardrail for Spotify vs Gmeet
        if best_label.lower() == "spotify" and packet_count > 1000:
            gmeet_dist = next((float(s["euclidean"]) for s in scores if "gmeet" in str(s["label"]).lower()), None)
            if gmeet_dist is not None:
                denom = max(gmeet_dist, 1e-12)
                if abs(gmeet_dist - best_distance) / denom < 0.20:
                    best_label = "Ambiguous (Likely Video Call)"

        if self.debug_verbose:
            second = scores[1] if len(scores) > 1 else None
            logging.debug(
                "Live distance ranking: best=%s l2=%.4f cosine=%.4f second=%s confidence=%.2f",
                best["label"],
                best_distance,
                best_cosine,
                second["label"] if second else "n/a",
                confidence,
            )

        return best_label, best_distance, confidence

    def _report_gui(self, msg_type: str, *args):
        # Pass to GUI if callback exists using non-blocking architecture
        if self.gui_callback:
            self.gui_callback(msg_type, *args)
            
        if msg_type == "inference":
            app_name, distance, packet_count, confidence = args
            msg = (
                f"[LIVE INFERENCE] DETECTED: {app_name} | Dist: {distance:.4f} | "
                f"Confidence: {confidence:.2f} | Window Packets: {packet_count}"
            )
            logging.info(msg)
            print(msg)

    def start(self):
        """Starts the multi-threaded pipeline."""
        if self.running:
            return
        self.running = True
        self.t_sniff = threading.Thread(target=self.sniff_worker, name="Sniffer")
        self.t_process = threading.Thread(target=self.process_worker, name="Processor")
        self.t_heartbeat = threading.Thread(target=self.heartbeat_worker, name="Heartbeat")
        
        self.t_sniff.daemon = True
        self.t_process.daemon = True
        self.t_heartbeat.daemon = True
        
        self.t_sniff.start()
        self.t_process.start()
        self.t_heartbeat.start()

    def stop(self):
        """Stops the pipeline."""
        if not self.running:
            return
        self.running = False
        logging.info("Stopping pipeline...")
        if self.t_process and self.t_process.is_alive():
            self.t_process.join(timeout=1.0)
        if self.t_heartbeat and self.t_heartbeat.is_alive():
            self.t_heartbeat.join(timeout=1.0)
        logging.info("Pipeline stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live VPN Inference Pipeline")
    parser.add_argument("--checkpoint", default="checkpoints/backbone_v1.pth", help="Path to .pth model")
    parser.add_argument("--library", default="templates/library.json", help="Path to reference library")
    parser.add_argument("--filter", default="", help="BPF Filter e.g. 'udp port 51820'")
    parser.add_argument("--interface", default=None, help="Capture interface")
    parser.add_argument(
        "--silence-threshold-bytes",
        type=int,
        default=DEFAULT_SILENCE_THRESHOLD_BYTES,
        help="Minimum bytes in a window before live inference is attempted.",
    )
    parser.add_argument(
        "--confidence-delta",
        type=float,
        default=CONFIDENCE_DELTA,
        help="Minimum percent gap between the top two Euclidean matches before reporting a specific app.",
    )
    args = parser.parse_args()
    
    pipeline = Phase2LivePipeline(
        checkpoint_path=args.checkpoint,
        library_path=args.library,
        bpf_filter=args.filter,
        interface=args.interface,
        silence_threshold_bytes=args.silence_threshold_bytes,
        confidence_delta=args.confidence_delta,
        window_size=60.0,
        stride=15.0,
        matrix_size=32 # 32x32 FlowPic
    )
    
    try:
        pipeline.start()
        print("Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
