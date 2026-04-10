from __future__ import annotations

import json
import threading
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


class FlowPicApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FlowPic VPN Fingerprinting")
        self.root.geometry("1100x760")
        self.preview_image: ImageTk.PhotoImage | None = None

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
        self.capture_filter = self._add_labeled_entry(live_frame, 2, "Capture Filter", "ip or ip6")
        
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

    def _refresh_interfaces(self) -> None:
        def action():
            interfaces = list_capture_interfaces()
            self.interface_mapping = {iface["name"]: iface["id"] for iface in interfaces}
            names = list(self.interface_mapping.keys())
            def update_ui():
                self.interface_combo["values"] = names
                if names:
                    self.interface_combo.current(0)
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
            row, _ = capture_and_generate_flowpic(
                output_image=self.capture_output.get(),
                output_pcap=self.capture_pcap.get() or None,
                duration=int(self.capture_duration.get()),
                interface=iface_id,
                bpf_filter=self.capture_filter.get() or None,
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


def main() -> None:
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    FlowPicApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
