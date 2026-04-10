import argparse
import csv
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

from flowpic.data_builder import build_dataset
from flowpic.library import register_application
from flowpic.predict import predict_input
from flowpic.train_pipeline import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_e2e_validation(
    raw_root: str | Path,
    epochs: int = 10,
    test_split: float = 0.2,
    threshold: float = 0.35,
) -> None:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        logging.error(f"Raw root directory '{raw_root}' does not exist.")
        return

    # Group PCAPs by label (parent directory name)
    pcaps_by_label = defaultdict(list)
    for pcap_path in raw_root.rglob("*.pcap"):
        label = pcap_path.parent.name
        pcaps_by_label[label].append(pcap_path)

    if not pcaps_by_label:
        logging.error(f"No PCAP files found in '{raw_root}'. Ensure files are in subdirectories named by label.")
        return

    # Split into train/test
    train_pcaps = []
    test_pcaps = []
    for label, pcaps in pcaps_by_label.items():
        if len(pcaps) < 2:
            logging.warning(f"Label '{label}' has only {len(pcaps)} PCAP(s). Cannot split. Adding to train set.")
            train_pcaps.extend(pcaps)
            continue
            
        random.shuffle(pcaps)
        split_idx = max(1, int(len(pcaps) * (1 - test_split)))
        train_pcaps.extend(pcaps[:split_idx])
        test_pcaps.extend(pcaps[split_idx:])

    logging.info(f"Split: {len(train_pcaps)} Train PCAPs, {len(test_pcaps)} Test PCAPs.")
    if not train_pcaps:
        logging.error("No training PCAPs available.")
        return
    if not test_pcaps:
        logging.error("No testing PCAPs available (not enough data to split). Add more PCAPs per label.")
        return

    # Setup temporary processing directories
    work_dir = Path("outputs/validation_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    train_raw_dir = work_dir / "train_raw"
    train_proc_dir = work_dir / "train_processed"
    manifest_path = work_dir / "train_manifest.csv"
    checkpoint_path = work_dir / "model.pth"
    library_path = work_dir / "library.json"

    # Copy train PCAPs into structured directory
    for pcap in train_pcaps:
        label = pcap.parent.name
        dest = train_raw_dir / label / pcap.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pcap, dest)

    # 1. Build Dataset
    logging.info("\n--- Phase 1: Building Dataset ---")
    rows = build_dataset(train_raw_dir, train_proc_dir, manifest_path)
    logging.info(f"Generated {len(rows)} FlowPics for training.")

    # 2. Train Model
    logging.info("\n--- Phase 2: Training Model ---")
    train_model(manifest_path, checkpoint_path, epochs=epochs, batch_size=8)
    logging.info("Training complete.")

    # 3. Register Templates
    logging.info("\n--- Phase 3: Registering App Templates ---")
    distinct_labels = {row.label for row in rows}
    for label in distinct_labels:
        logging.info(f"Registering template for: {label}")
        register_application(label, checkpoint_path, library_path, manifest_path)
    
    # 4. Evaluate Test Set
    logging.info("\n--- Phase 4: Evaluating on Test Set ---")
    correct = 0
    total = len(test_pcaps)
    predictions_log = []

    for test_pcap in test_pcaps:
        true_label = test_pcap.parent.name
        try:
            result = predict_input(
                checkpoint_path=checkpoint_path,
                library_path=library_path,
                pcap_path=test_pcap,
                threshold=threshold
            )
            pred_label = result["prediction"]
            distance = result["distance"]
        except Exception as e:
            logging.warning(f"Failed to predict PCAP {test_pcap.name}: {e}")
            pred_label = "error"
            distance = None
            
        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1
            
        predictions_log.append({
            "pcap": test_pcap.name,
            "true": true_label,
            "pred": pred_label,
            "distance": distance,
            "correct": is_correct
        })
        
        status = "HIT " if is_correct else "MISS"
        logging.info(f"[{status}] File: {test_pcap.name:20s} | True: {true_label:15s} | Pred: {pred_label:15s} | Dist: {distance}")

    accuracy = correct / total * 100
    logging.info("\n--- Validation Summary ---")
    logging.info(f"Total Test Files: {total}")
    logging.info(f"Correct: {correct}")
    logging.info(f"Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Validation on Real PCAPs.")
    parser.add_argument("--data", default="data/raw", help="Path to raw PCAP directory structure.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of PCAPs to use for testing.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Anomaly threshold for 'Unknown'.")

    args = parser.parse_args()
    run_e2e_validation(
        raw_root=args.data,
        epochs=args.epochs,
        test_split=args.test_split,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
