from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from .capture import capture_and_generate_flowpic, list_capture_interfaces
from .data_builder import build_dataset
from .demo import create_demo_assets, run_demo
from .generate import generate_flowpic_preview
from .library import register_application
from .predict import predict_input
from .train_pipeline import train_model
from .live_capture_phase2 import Phase2LivePipeline


APP_CATEGORIES = {
    "youtube_4k": "YouTube (4K)",
    "youtube_720p": "YouTube (HD)",
    "spotify": "Spotify/Audio",
    "gmeet_hd": "Google Meet",
    "idle_noise": "System Idle"
}

BPF_PRESETS = {
    "Standard Web (No VPN) [TCP/UDP 443]": "(tcp port 443 or udp port 443)",
    "IPSec / Disguised [UDP 4500]": "udp port 4500 or udp port 500 or esp",
    "WireGuard [UDP 51820]": "udp port 51820",
    "Google Meet (Advanced)": "udp portrange 19302-19309 or udp portrange 10000-20000 or port 443",
    "All IPv4/IPv6 Traffic (TUN Interface)": "ip or ip6",
    "Custom BPF Filter...": ""
}

class FlowPicApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FlowPic VPN Fingerprinting")
        self.root.geometry("1100x760")
        self.preview_image: ImageTk.PhotoImage | None = None
        self.live_buffer = __import__('collections').deque(maxlen=5)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        self.log = scrolledtext.ScrolledText(root, height=10)
        self.log.pack(fill="both", expand=False, padx=10, pady=5)

        self.preview_label = ttk.Label(root, text="Latest FlowPic preview will appear here.", anchor="center")
        self.preview_label.pack(fill="both", expand=False, padx=10, pady=5)

        self.status_frame = ttk.Frame(root)
        self.status_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))
        
        self.status_label = ttk.Label(self.status_frame, text="Ready.")
        self.status_label.pack(side="left")
        
        self.progress = ttk.Progressbar(self.status_frame, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(10, 0))

        self._build_flowpic_tab()
        self._build_dataset_tab()
        self._build_training_tab()
        self._build_library_tab()
        self._build_demo_tab()
        self._build_live_inference_tab()
        
        self.live_pipeline = None

    def _add_labeled_entry(self, parent, row: int, label: str, default: str = "") -> tk.StringVar:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=variable, width=70).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        return variable

    def _add_labeled_combobox(self, parent, row: int, label: str) -> tuple[ttk.Combobox, tk.StringVar]:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar()
        combo = ttk.Combobox(parent, textvariable=variable, width=67, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        return combo, variable

    def _choose_file(self, variable: tk.StringVar, *, save: bool = False, directory: bool = False) -> None:
        if directory:
            result = filedialog.askdirectory()
        elif save:
            result = filedialog.asksaveasfilename(defaultextension=".png")
        else:
            result = filedialog.askopenfilename()
        if result:
            variable.set(result)

    def _run_async(self, target, success_message: str | None = None, format_func=None) -> None:
        def runner():
            self.root.after(0, self._start_progress)
            try:
                result = target()
                if success_message:
                    self._append_log(success_message)
                if result is not None:
                    if format_func:
                        self._append_log(format_func(result))
                    else:
                        self._append_log(json.dumps(result, indent=2, default=str))
            except Exception as exc:  # pragma: no cover - GUI path
                self.root.after(0, lambda: messagebox.showerror("FlowPic Error", str(exc)))
                self._append_log(f"ERROR: {exc}")
            finally:
                self.root.after(0, self._stop_progress)

        threading.Thread(target=runner, daemon=True).start()

    def _start_progress(self) -> None:
        self.log.delete("1.0", "end")
        self.progress.start(10)
        self.status_label.configure(text="Processing, please wait...")
        # Disable tabs superficially
        self.notebook.state(["disabled"])

    def _stop_progress(self) -> None:
        self.progress.stop()
        self.status_label.configure(text="Ready.")
        self.notebook.state(["!disabled"])

    def _append_log(self, message: str) -> None:
        self.root.after(0, lambda: self._write_log(message))

    def _write_log(self, message: str) -> None:
        self.log.insert("end", f"{message}\n")
        self.log.see("end")

    def _set_preview(self, image_path: str | Path) -> None:
        def update():
            image = Image.open(image_path).convert("L")
            image.thumbnail((420, 420))
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_image, text="")

        self.root.after(0, update)

    def _build_flowpic_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        self.notebook.add(frame, text="FlowPic Setup")
        self.interface_mapping = {}

        live_frame = ttk.LabelFrame(frame, text="Live Capture", padding=10)
        live_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        live_frame.columnconfigure(1, weight=1)

        self.interface_combo, self.capture_interface = self._add_labeled_combobox(live_frame, 0, "Interface")
        self.capture_duration = self._add_labeled_entry(live_frame, 1, "Duration (s)", "15")
        
        ttk.Label(live_frame, text="Capture Filter").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        
        self.record_filter_friendly = tk.StringVar(value="All IPv4/IPv6 Traffic (TUN Interface)")
        self.capture_filter = tk.StringVar(value="ip or ip6")
        
        self.combo_record_filter = ttk.Combobox(
            live_frame, 
            textvariable=self.record_filter_friendly, 
            width=67,
            values=list(BPF_PRESETS.keys()),
            state="readonly"
        )
        self.combo_record_filter.grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        
        def on_record_preset_change(event):
            selected = self.record_filter_friendly.get()
            if selected == "Custom BPF Filter...":
                self.combo_record_filter.config(state="normal")
            else:
                self.combo_record_filter.config(state="readonly")
                self.capture_filter.set(BPF_PRESETS.get(selected, "ip or ip6"))
                
                # Auto-Filename Suffix Logic
                is_vpn = "WireGuard" in selected or "IPSec" in selected
                out_png = Path(self.capture_output.get())
                out_pcap = Path(self.capture_pcap.get())
                
                if is_vpn and not out_png.stem.endswith("_vpn"):
                    self.capture_output.set(str(out_png.with_name(out_png.stem + "_vpn" + out_png.suffix)))
                    self.capture_pcap.set(str(out_pcap.with_name(out_pcap.stem + "_vpn" + out_pcap.suffix)))
                elif not is_vpn and out_png.stem.endswith("_vpn"):
                    self.capture_output.set(str(out_png.with_name(out_png.stem[:-4] + out_png.suffix)))
                    self.capture_pcap.set(str(out_pcap.with_name(out_pcap.stem[:-4] + out_pcap.suffix)))

        self.combo_record_filter.bind("<<ComboboxSelected>>", on_record_preset_change)
        
        btn_frame = ttk.Frame(live_frame)
        btn_frame.grid(row=3, column=1, sticky="w", padx=6, pady=8)
        ttk.Button(btn_frame, text="Refresh Interfaces", command=self._refresh_interfaces).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Test Interface (2s)", command=self._test_interface).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Capture & View", command=self._capture_live).pack(side="left")

        pcap_frame = ttk.LabelFrame(frame, text="Offline PCAP Processing", padding=10)
        pcap_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        pcap_frame.columnconfigure(1, weight=1)

        self.pcap_input = self._add_labeled_entry(pcap_frame, 0, "Source PCAP", "")
        self.pcap_window_index = self._add_labeled_entry(pcap_frame, 1, "Window Index (-1=auto)", "-1")
        ttk.Button(pcap_frame, text="Browse", command=lambda: self._choose_file(self.pcap_input)).grid(row=0, column=2, padx=6)

        btn_frame_pcap = ttk.Frame(pcap_frame)
        btn_frame_pcap.grid(row=2, column=1, sticky="w", padx=6, pady=8)
        ttk.Button(btn_frame_pcap, text="Generate FlowPic", command=self._generate_from_pcap).pack(side="left")

        out_frame = ttk.LabelFrame(frame, text="Outputs", padding=10)
        out_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        out_frame.columnconfigure(1, weight=1)
        self.capture_output = self._add_labeled_entry(out_frame, 0, "PNG Output Path", str(Path("outputs/live_preview.png")))
        self.capture_pcap = self._add_labeled_entry(out_frame, 1, "PCAP Output Path", str(Path("outputs/live_preview.pcap")))
        ttk.Button(out_frame, text="Browse", command=lambda: self._choose_file(self.capture_output, save=True)).grid(row=0, column=2, padx=6)
        ttk.Button(out_frame, text="Browse", command=lambda: self._choose_file(self.capture_pcap, save=True)).grid(row=1, column=2, padx=6)
        
        self.root.after(200, self._refresh_interfaces)

    def _build_dataset_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        self.notebook.add(frame, text="Dataset")

        self.raw_root = self._add_labeled_entry(frame, 0, "Raw Root", str(Path("data/raw")))
        self.processed_root = self._add_labeled_entry(frame, 1, "Processed Root", str(Path("data/processed")))
        self.manifest_path = self._add_labeled_entry(frame, 2, "Manifest", str(Path("data/processed/manifest.csv")))

        ttk.Button(frame, text="Raw Folder", command=lambda: self._choose_file(self.raw_root, directory=True)).grid(row=0, column=2, padx=6)
        ttk.Button(frame, text="Processed Folder", command=lambda: self._choose_file(self.processed_root, directory=True)).grid(row=1, column=2, padx=6)
        ttk.Button(frame, text="Manifest File", command=lambda: self._choose_file(self.manifest_path, save=True)).grid(row=2, column=2, padx=6)
        ttk.Button(frame, text="Build Dataset", command=self._build_dataset_action).grid(row=3, column=1, padx=6, pady=10, sticky="w")

    def _build_training_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        self.notebook.add(frame, text="Training")

        self.train_manifest = self._add_labeled_entry(frame, 0, "Manifest", str(Path("data/processed/manifest.csv")))
        self.checkpoint_path = self._add_labeled_entry(frame, 1, "Checkpoint", str(Path("checkpoints/backbone_v1.pth")))
        self.train_epochs = self._add_labeled_entry(frame, 2, "Epochs", "10")
        self.train_batch_size = self._add_labeled_entry(frame, 3, "Batch Size", "8")

        ttk.Button(frame, text="Choose Manifest", command=lambda: self._choose_file(self.train_manifest)).grid(row=0, column=2, padx=6)
        ttk.Button(frame, text="Choose Checkpoint", command=lambda: self._choose_file(self.checkpoint_path, save=True)).grid(row=1, column=2, padx=6)
        ttk.Button(frame, text="Train Model", command=self._train_model_action).grid(row=4, column=1, padx=6, pady=10, sticky="w")

    def _build_library_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        self.notebook.add(frame, text="Library + Predict")

        self.library_checkpoint = self._add_labeled_entry(frame, 0, "Checkpoint", str(Path("checkpoints/backbone_v1.pth")))
        self.library_file = self._add_labeled_entry(frame, 1, "Library JSON", str(Path("templates/library.json")))
        self.register_label = self._add_labeled_entry(frame, 2, "Register Label", "")
        self.register_manifest = self._add_labeled_entry(frame, 3, "Register Manifest", str(Path("data/processed/manifest.csv")))
        self.predict_image = self._add_labeled_entry(frame, 4, "Predict Image", "")
        self.predict_pcap = self._add_labeled_entry(frame, 5, "Predict PCAP", "")
        self.predict_threshold = self._add_labeled_entry(frame, 6, "Threshold", "0.35")

        ttk.Button(frame, text="Checkpoint", command=lambda: self._choose_file(self.library_checkpoint)).grid(row=0, column=2, padx=6)
        ttk.Button(frame, text="Library File", command=lambda: self._choose_file(self.library_file, save=True)).grid(row=1, column=2, padx=6)
        ttk.Button(frame, text="Manifest", command=lambda: self._choose_file(self.register_manifest)).grid(row=3, column=2, padx=6)
        ttk.Button(frame, text="Image", command=lambda: self._choose_file(self.predict_image)).grid(row=4, column=2, padx=6)
        ttk.Button(frame, text="PCAP", command=lambda: self._choose_file(self.predict_pcap)).grid(row=5, column=2, padx=6)
        ttk.Button(frame, text="Register App", command=self._register_application_action).grid(row=7, column=1, padx=6, pady=10, sticky="w")
        ttk.Button(frame, text="Predict Unknown", command=self._predict_unknown_action).grid(row=7, column=2, padx=6, pady=10, sticky="w")

    def _build_demo_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        self.notebook.add(frame, text="Demo")

        self.demo_root = self._add_labeled_entry(frame, 0, "Demo Root", str(Path("demo_assets")))
        self.demo_epochs = self._add_labeled_entry(frame, 1, "Demo Epochs", "6")
        self.demo_threshold = self._add_labeled_entry(frame, 2, "Demo Threshold", "0.08")

        ttk.Button(frame, text="Choose Folder", command=lambda: self._choose_file(self.demo_root, directory=True)).grid(row=0, column=2, padx=6)
        ttk.Label(
            frame,
            text=(
                "This tab creates a synthetic visual demo. It trains on known apps, predicts a known sample, "
                "flags a zero-day sample as Unknown, then registers that new app into the template library "
                "without retraining and predicts it again."
            ),
            wraplength=760,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(10, 12))

        ttk.Button(frame, text="Create Demo Assets", command=self._create_demo_assets_action).grid(row=4, column=0, padx=6, pady=8, sticky="w")
        ttk.Button(frame, text="Run Full Demo", command=self._run_demo_action).grid(row=4, column=1, padx=6, pady=8, sticky="w")

    def _build_live_inference_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        self.notebook.add(frame, text="Live Inference (Phase 2)")

        self.live_iface_combo, self.live_capture_interface = self._add_labeled_combobox(frame, 0, "Interface")
        
        ttk.Label(frame, text="VPN Protocol / BPF").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        
        self.live_filter_friendly = tk.StringVar(value="Select a Preset...")
        self.live_filter = tk.StringVar(value="ip or ip6")
        
        filter_combo = ttk.Combobox(
            frame, 
            textvariable=self.live_filter_friendly, 
            width=67,
            values=list(BPF_PRESETS.keys()),
            state="readonly"
        )
        filter_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        
        def on_preset_change(event):
            self.live_buffer.clear()
            selected = self.live_filter_friendly.get()
            if selected == "Custom BPF Filter...":
                # Allow manual entry if they pick custom
                filter_combo.config(state="normal")
            else:
                filter_combo.config(state="readonly")
                self.live_filter.set(BPF_PRESETS.get(selected, "ip or ip6"))
        
        filter_combo.bind("<<ComboboxSelected>>", on_preset_change)
        
        # Add a small hint label for the raw BPF string
        self.bpf_hint = ttk.Label(frame, textvariable=self.live_filter, font=("Courier", 8), foreground="gray")
        self.bpf_hint.grid(row=2, column=1, sticky="w", padx=6)
        
        self.live_checkpoint = self._add_labeled_entry(frame, 3, "Checkpoint", str(Path("checkpoints/backbone_v1.pth")))
        self.live_library = self._add_labeled_entry(frame, 4, "Library JSON", str(Path("templates/library.json")))
        self.live_silence = self._add_labeled_entry(frame, 5, "Silence Threshold (Bytes)", "20000")
        
        ttk.Button(frame, text="Refresh", command=self._refresh_interfaces).grid(row=0, column=2, padx=6)
        ttk.Button(frame, text="Browse", command=lambda: self._choose_file(self.live_checkpoint)).grid(row=3, column=2, padx=6)
        ttk.Button(frame, text="Browse", command=lambda: self._choose_file(self.live_library)).grid(row=4, column=2, padx=6)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=1, sticky="w", padx=6, pady=10)
        self.btn_live_start = ttk.Button(btn_frame, text="Start Live Inference", command=self._start_live_inference)
        self.btn_live_start.pack(side="left", padx=(0, 6))
        self.btn_live_stop = ttk.Button(btn_frame, text="Stop Live Inference", command=self._stop_live_inference, state="disabled")
        self.btn_live_stop.pack(side="left")

        # Results area
        self.live_status_label = ttk.Label(frame, text="Smoothed Result: None", font=("Helvetica", 14, "bold"))
        self.live_status_label.grid(row=7, column=0, columnspan=3, sticky="w", pady=(10, 0), padx=6)

        self.live_results = scrolledtext.ScrolledText(frame, height=12)
        self.live_results.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(5, 10))
        frame.rowconfigure(8, weight=1)

    def _refresh_interfaces(self) -> None:
        def action():
            interfaces = list_capture_interfaces()
            self.interface_mapping = {iface["name"]: iface["id"] for iface in interfaces}
            names = list(self.interface_mapping.keys())
            def update_ui():
                self.interface_combo["values"] = names
                self.live_iface_combo["values"] = names
                if names:
                    self.interface_combo.current(0)
                    self.live_iface_combo.current(0)
            self.root.after(0, update_ui)
            return {"found_interfaces": len(interfaces)}
        self._run_async(action, success_message="Interfaces refreshed.")

    def _test_interface(self) -> None:
        def action():
            from .capture import capture_packets_to_pcap
            iface_name = self.capture_interface.get()
            iface_id = self.interface_mapping.get(iface_name) or iface_name or None
            self._append_log(f"\nTesting interface '{iface_name}' for 2 seconds...")
            out_pcap = Path("outputs/test_capture.pcap")
            out_pcap.parent.mkdir(exist_ok=True, parents=True)
            packets = capture_packets_to_pcap(
                output_pcap=out_pcap,
                duration=2,
                interface=iface_id,
                bpf_filter=self.capture_filter.get() or None,
            )
            return {"interface": iface_name, "packets_captured": len(packets)}
        self._run_async(action, success_message="Interface test complete.")

    def _capture_live(self) -> None:
        def action():
            iface_name = self.capture_interface.get()
            iface_id = self.interface_mapping.get(iface_name) or iface_name or None
            self._append_log(f"\nStarting live capture on '{iface_name}'...")
            
            selected_preset = self.record_filter_friendly.get()
            final_bpf = BPF_PRESETS.get(selected_preset, selected_preset)
            if selected_preset == "Custom BPF Filter...":
                final_bpf = ""
                
            row, _ = capture_and_generate_flowpic(
                output_image=self.capture_output.get(),
                output_pcap=self.capture_pcap.get() or None,
                duration=int(self.capture_duration.get()),
                interface=iface_id,
                bpf_filter=final_bpf or None,
                show=False,
            )
            self._set_preview(row.image_path)
            return row.__dict__

        self._run_async(action, success_message="Live capture completed.")

    def _generate_from_pcap(self) -> None:
        def action():
            row = generate_flowpic_preview(
                pcap_path=self.pcap_input.get(),
                output_path=self.capture_output.get(),
                manifest_path=Path(self.capture_output.get()).with_suffix(".csv"),
                show=False,
                window_index=int(self.pcap_window_index.get()),
            )
            self._set_preview(row.image_path)
            return row.__dict__

        self._run_async(action, success_message="FlowPic generated from PCAP.")

    def _build_dataset_action(self) -> None:
        def formatter(res):
            return f"\n---> Dataset generation successful! Generated {res['num_rows']} FlowPic slices.\n"

        self._run_async(
            lambda: {
                "num_rows": len(
                    build_dataset(
                        raw_root=self.raw_root.get(),
                        processed_root=self.processed_root.get(),
                        manifest_path=self.manifest_path.get(),
                    )
                )
            },
            success_message="Dataset build completed.",
            format_func=formatter
        )

    def _train_model_action(self) -> None:
        def formatter(res):
            return (
                f"\n=== TRAINING COMPLETE ===\n"
                f"Classes Learned : {res.get('num_classes', '?')}\n"
                f"Total Samples   : {res.get('num_samples', '?')}\n"
                f"Final Loss Score: {res.get('best_loss', 0.0):.4f}\n"
                f"Model saved to  : {res.get('checkpoint_path', '?')}\n"
                f"=========================\n"
            )

        self._run_async(
            lambda: train_model(
                self.train_manifest.get(),
                self.checkpoint_path.get(),
                epochs=int(self.train_epochs.get()),
                batch_size=int(self.train_batch_size.get()),
            ),
            success_message="Training completed.",
            format_func=formatter
        )

    def _register_application_action(self) -> None:
        self._run_async(
            lambda: register_application(
                label=self.register_label.get(),
                checkpoint_path=self.library_checkpoint.get(),
                library_path=self.library_file.get(),
                manifest_path=self.register_manifest.get(),
            ),
            success_message="Application template registered.",
        )

    def _predict_unknown_action(self) -> None:
        def action():
            result = predict_input(
                checkpoint_path=self.library_checkpoint.get(),
                library_path=self.library_file.get(),
                image_path=self.predict_image.get() or None,
                pcap_path=self.predict_pcap.get() or None,
                threshold=float(self.predict_threshold.get()),
            )
            if str(result.get("input", "")).lower().endswith(".png"):
                self._set_preview(result["input"])
            return result

        def formatter(res):
            status = "UNKNOWN (Anomaly)" if res.get('is_unknown') else "KNOWN"
            return (
                f"\n=== PREDICTION RESULT ===\n"
                f"File          : {Path(res['input']).name}\n"
                f"Predicted App : {res['label']} ({status})\n"
                f"Distance/Conf : {res['distance']:.4f} (Threshold: {res['threshold']})\n"
                f"=========================\n"
            )

        self._run_async(action, success_message="Prediction completed.", format_func=formatter)

    def _create_demo_assets_action(self) -> None:
        self._run_async(
            lambda: create_demo_assets(self.demo_root.get()),
            success_message="Demo assets created.",
        )

    def _run_demo_action(self) -> None:
        def action():
            result = run_demo(
                self.demo_root.get(),
                epochs=int(self.demo_epochs.get()),
                threshold=float(self.demo_threshold.get()),
            )
            preview_path = Path(result["zero_day_after_registration"]["input"])
            if preview_path.exists():
                self._set_preview(preview_path)
            return result

        self._run_async(action, success_message="Full demo pipeline completed.")

    def _start_live_inference(self) -> None:
        if self.live_pipeline and self.live_pipeline.running:
            return
            
        iface_name = self.live_capture_interface.get()
        iface_id = self.interface_mapping.get(iface_name) or iface_name or None
        
        def gui_callback(msg_type, *args):
            if msg_type == "inference":
                app_name, distance, packet_count, confidence = args
                
                # Rule 5: Volatility Reset
                if len(self.live_buffer) >= 1:
                    last_3 = list(self.live_buffer)[-3:]
                    avg_packets = sum(item["packet_count"] for item in last_3) / len(last_3)
                    if avg_packets > 0 and packet_count < (0.3 * avg_packets):
                        self.live_buffer.clear()
                
                # Rule 4: Handle "Heavy Idle" Problem
                if app_name == "idle_noise" and packet_count > 1000:
                    app_name = "unknown"
                    
                # Rule 6: Generic Streaming becomes Idle/Browsing
                if "Generic" in app_name:
                    app_name = "idle_browsing"
                    
                self.live_buffer.append({
                    "app_name": app_name,
                    "distance": distance,
                    "packet_count": packet_count,
                    "confidence": confidence
                })
            else:
                text = args[0]
                # Rule 1: Reset on Silence
                if "Filtering Noise" in text:
                    self.live_buffer.clear()

            # Default if buffer is cleared by Rule 1
            smoothed_category = APP_CATEGORIES.get("idle_noise") if not self.live_buffer else "None"

            if self.live_buffer:
                from collections import Counter
                labels = [item["app_name"] for item in self.live_buffer]
                most_common = Counter(labels).most_common(1)
                best_lbl = most_common[0][0] if most_common else "None"
                
                smoothed_category = APP_CATEGORIES.get(str(best_lbl).lower(), f"🌐 {best_lbl}")

            if msg_type == "inference":
                msg = (
                    f"[{time.strftime('%H:%M:%S')}] RAW: {self.live_buffer[-1]['app_name']} | "
                    f"Dist: {args[1]:.4f} | Confidence: {args[3]:.2f} | Packets: {args[2]}\n"
                )
            else:
                msg = f"[{time.strftime('%H:%M:%S')}] {args[0]}\n"

            def update_gui():
                self.live_status_label.config(text=f"Smoothed Result: {smoothed_category}")
                self.live_results.insert("end", msg)
                self.live_results.see("end")

            self.root.after(0, update_gui)

        try:
            silence_val = int(self.live_silence.get() or "50000")
            
            # Use the typed text if it's a custom filter, otherwise use the mapped preset
            selected_preset = self.live_filter_friendly.get()
            final_bpf = BPF_PRESETS.get(selected_preset, selected_preset)
            if selected_preset == "Custom BPF Filter...":
                final_bpf = "" # If they left it as exactly "Custom BPF Filter..."
            
            self.live_pipeline = Phase2LivePipeline(
                checkpoint_path=self.live_checkpoint.get(),
                library_path=self.live_library.get(),
                interface=iface_id,
                bpf_filter=final_bpf,
                silence_threshold_bytes=silence_val,
                gui_callback=gui_callback
            )
            self.live_pipeline.start()
            self.btn_live_start.config(state="disabled")
            self.btn_live_stop.config(state="normal")
            self.live_results.delete("1.0", "end")
            self.live_results.insert("end", "=== Started Live Inference (5s Window, 2s Stride) ===\n")
            self._append_log("Started Live Inference Pipeline.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _stop_live_inference(self) -> None:
        if self.live_pipeline:
            self.live_pipeline.stop()
        self.btn_live_start.config(state="normal")
        self.btn_live_stop.config(state="disabled")
        self.live_results.insert("end", "=== Stopped Live Inference ===\n")
        self._append_log("Stopped Live Inference Pipeline.")


def main() -> None:
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    FlowPicApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
