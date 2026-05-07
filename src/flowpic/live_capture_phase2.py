import threading
import time
import logging
from collections import deque
from pathlib import Path
import numpy as np
import torch

from .library import load_model_checkpoint
from .predict import load_library_tensors
from .matching import match_embedding

try:
    from scapy.all import sniff
except ImportError:
    raise ImportError("Scapy is required. Please install it using 'pip install scapy'")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    datefmt='%H:%M:%S'
)

class Phase2LivePipeline:
    """
    Phase-2 Pipeline for VPN Identification: Live Packet Capture and FlowPic Inference.
    
    Features:
    - Multi-threaded: High-speed sniffing + separate processing thread.
    - True sliding window using high-precision time (time_ns).
    - Map IAT and Size to 2D Matrix (32x32 or 64x64).
    - Pixel Normalization 0-255 based on packet density.
    - Live Inference Pipeline (CNN -> 128-d embedding -> Euclidean Distance against Library).
    - Graceful handling of silent periods.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        library_path: str | Path,
        interface: str = None,
        bpf_filter: str = "udp port 51820",
        window_size: float = 5.0,
        stride: float = 2.0,
        silence_threshold_bytes: int = 1000,
        max_mtu: int = 1600,
        matrix_size: int = 224, # Must match Phase 1 training (224x224)
        threshold: float = 0.35,
        gui_callback=None,
        debug_verbose: bool = True
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.window_size = window_size
        self.stride = stride
        self.silence_threshold_bytes = silence_threshold_bytes
        self.max_mtu = max_mtu
        self.bins = matrix_size
        self.threshold = threshold
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
                msg = f"[i] Window complete: Only {total_bytes}/{self.silence_threshold_bytes} bytes captured. Skipping inference."
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
        
        # 3. Log-Contrast & Percentile Normalization (Phase 1 Logic)
        matrix = np.log1p(matrix)
        nonzero = matrix[matrix > 0]
        if nonzero.size > 0:
            scale = float(np.quantile(nonzero, 0.995))
            if scale <= 0:
                scale = float(nonzero.max(initial=0.0))
            if scale > 0:
                matrix = np.clip(matrix / scale, 0.0, 1.0)
                
        matrix = matrix.T # Shape: (224, 224)
        
        # Matrix "Sanity Check"
        if matrix.max() == 0:
            error_msg = "[!] Error: Generated FlowPic is empty despite packet count."
            logging.error(error_msg)
            self._report_gui("log", error_msg)
            return
            
        # 4. Live Inference Pipeline
        # Convert to PyTorch Tensor: shape [1, 1, 224, 224]
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Pass through CNN to extract 128-d embedding
            embedding = self.model(tensor)
            
        # 5. Compare using Cosine Distance (Matches Phase 1's 0.35 Threshold)
        if not self.template_library:
            self._report_gui("inference", "Unknown (No Library)", float('inf'), len(packets))
            return
            
        result = match_embedding(
            embedding, 
            self.template_library, 
            threshold=self.threshold, 
            metric="cosine"
        )
        
        # 6. Output Application Name and Confidence Score
        self._report_gui("inference", result.label, result.distance, len(packets))

    def _report_gui(self, msg_type: str, *args):
        # Pass to GUI if callback exists using non-blocking architecture
        if self.gui_callback:
            self.gui_callback(msg_type, *args)
            
        if msg_type == "inference":
            app_name, distance, packet_count = args
            msg = f"[LIVE INFERENCE] App: {app_name} | Distance: {distance:.4f} | Window Packets: {packet_count}"
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
    args = parser.parse_args()
    
    pipeline = Phase2LivePipeline(
        checkpoint_path=args.checkpoint,
        library_path=args.library,
        bpf_filter=args.filter,
        interface=args.interface,
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
