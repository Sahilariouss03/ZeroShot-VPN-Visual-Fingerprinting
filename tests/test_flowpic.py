from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from unittest.mock import patch

from flowpic.capture import capture_and_generate_flowpic, list_capture_interfaces
from flowpic.data_builder import build_dataset
from flowpic.demo import create_demo_assets, run_demo
from flowpic.generate import build_flowpic_histogram, save_flowpic_image, write_manifest
from flowpic.library import load_template_library, register_application
from flowpic.matching import build_template_library, match_embedding
from flowpic.model import FlowPicEmbeddingNet
from flowpic.predict import predict_input
from flowpic.preprocess import FlowWindow, PacketRecord, packet_record_from_scapy_packet, window_packet_records
from flowpic.train import build_triplet_loss, train_triplet_step
from flowpic.train_pipeline import train_model


def _records() -> list[PacketRecord]:
    return [
        PacketRecord(timestamp=10.0, relative_time=0.0, inter_arrival_time=0.0, packet_size=120, direction=1),
        PacketRecord(timestamp=10.2, relative_time=0.2, inter_arrival_time=0.2, packet_size=400, direction=1),
        PacketRecord(timestamp=11.0, relative_time=1.0, inter_arrival_time=0.8, packet_size=900, direction=-1),
        PacketRecord(timestamp=14.4, relative_time=4.4, inter_arrival_time=3.4, packet_size=1500, direction=-1),
    ]


def test_histogram_generation_and_manifest(tmp_path: Path) -> None:
    window = FlowWindow(index=0, start_time=0.0, end_time=5.0, actual_end_time=4.4, packets=_records())
    histogram = build_flowpic_histogram(window, bins_time=224, bins_size=224, time_feature="arrival_time")
    assert histogram.shape == (224, 224)
    assert float(histogram.sum()) > 0.0

    output_path = tmp_path / "preview.png"
    save_flowpic_image(histogram, output_path)
    assert output_path.exists()

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        [
            {
                "label": "demo",
                "source_pcap": "sample.pcap",
                "window_index": 0,
                "window_start": 0.0,
                "window_end": 5.0,
                "packet_count": 4,
                "image_path": str(output_path),
            }
        ],
        manifest_path,
    )
    assert manifest_path.exists()


def test_arrival_and_inter_arrival_modes_are_valid() -> None:
    window = FlowWindow(index=0, start_time=0.0, end_time=5.0, actual_end_time=4.4, packets=_records())
    arrival_hist = build_flowpic_histogram(window, bins_time=32, bins_size=32, time_feature="arrival_time")
    iat_hist = build_flowpic_histogram(window, bins_time=32, bins_size=32, time_feature="inter_arrival")

    assert arrival_hist.shape == (32, 32)
    assert iat_hist.shape == (32, 32)
    assert torch.isfinite(torch.from_numpy(arrival_hist)).all()
    assert torch.isfinite(torch.from_numpy(iat_hist)).all()


def test_short_window_is_retained_only_when_it_is_the_only_window() -> None:
    short_records = [
        PacketRecord(timestamp=1.0, relative_time=0.0, inter_arrival_time=0.0, packet_size=100, direction=1),
        PacketRecord(timestamp=1.2, relative_time=0.2, inter_arrival_time=0.2, packet_size=150, direction=1),
    ]
    single_window = window_packet_records(short_records, window_seconds=5.0, allow_single_short_window=True)
    assert len(single_window) == 1

    two_window_records = [
        PacketRecord(timestamp=1.0, relative_time=0.0, inter_arrival_time=0.0, packet_size=100, direction=1),
        PacketRecord(timestamp=2.0, relative_time=2.0, inter_arrival_time=1.0, packet_size=140, direction=1),
        PacketRecord(timestamp=5.2, relative_time=5.2, inter_arrival_time=3.2, packet_size=180, direction=1),
        PacketRecord(timestamp=9.9, relative_time=9.9, inter_arrival_time=4.7, packet_size=200, direction=-1),
        PacketRecord(timestamp=10.1, relative_time=10.1, inter_arrival_time=0.2, packet_size=250, direction=-1),
    ]
    multiple_windows = window_packet_records(two_window_records, window_seconds=5.0, allow_single_short_window=True)
    assert len(multiple_windows) == 2
    assert [window.index for window in multiple_windows] == [0, 1]


def test_model_outputs_normalized_128d_embeddings() -> None:
    model = FlowPicEmbeddingNet()
    inputs = torch.randn(4, 1, 224, 224)
    embeddings = model(inputs)

    assert embeddings.shape == (4, 128)
    norms = torch.norm(embeddings, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_triplet_training_step_reduces_loss() -> None:
    torch.manual_seed(7)
    model = FlowPicEmbeddingNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_triplet_loss(margin=0.2)

    base_a = torch.zeros(1, 224, 224)
    base_b = torch.ones(1, 224, 224)
    inputs = torch.stack(
        [
            base_a,
            base_a + 0.05 * torch.randn(1, 224, 224),
            base_b,
            base_b + 0.05 * torch.randn(1, 224, 224),
        ]
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    losses = [train_triplet_step(model, optimizer, loss_fn, inputs, labels) for _ in range(5)]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] <= losses[0]


def test_matching_returns_known_label_or_unknown() -> None:
    template_library = build_template_library(
        {
            "YouTube": torch.tensor([[1.0, 0.0], [0.9, 0.1]], dtype=torch.float32),
            "Zoom": torch.tensor([[0.0, 1.0], [0.1, 0.9]], dtype=torch.float32),
        }
    )

    known = match_embedding(torch.tensor([0.95, 0.05]), template_library, threshold=0.2, metric="cosine")
    unknown = match_embedding(torch.tensor([0.7, 0.7]), template_library, threshold=0.01, metric="cosine")

    assert known.label == "YouTube"
    assert not known.is_unknown
    assert unknown.label == "Unknown"
    assert unknown.is_unknown


def test_packet_record_from_packet_filters_and_normalizes() -> None:
    class FakeLayer:
        def __init__(self, sport: int | None = None, dport: int | None = None, src: str = "", dst: str = "") -> None:
            self.sport = sport
            self.dport = dport
            self.src = src
            self.dst = dst

    class FakePacket:
        def __init__(self) -> None:
            self.time = 10.5
            self.layers = {
                "IP": FakeLayer(src="10.0.0.1", dst="10.0.0.2"),
                "TCP": FakeLayer(sport=443, dport=53000),
            }

        def haslayer(self, layer) -> bool:
            return layer.__name__ in self.layers

        def getlayer(self, layer):
            return self.layers.get(layer.__name__)

        def __len__(self) -> int:
            return 512

    FakeIP = type("IP", (), {})
    FakeIPv6 = type("IPv6", (), {})
    FakeTCP = type("TCP", (), {})
    FakeUDP = type("UDP", (), {})

    with patch("flowpic.preprocess._load_scapy", return_value=(object(), FakeIP, FakeIPv6, FakeTCP, FakeUDP)):
        record = packet_record_from_scapy_packet(FakePacket(), first_timestamp=10.0, previous_timestamp=10.2)

    assert record is not None
    assert record.relative_time == 0.5
    assert record.inter_arrival_time == pytest.approx(0.3)
    assert record.packet_size == 512
    assert record.direction == 1


def test_capture_and_generate_flowpic_reuses_generator(tmp_path: Path) -> None:
    output_path = tmp_path / "live.png"
    expected_row = {
        "label": "live_capture",
        "source_pcap": str(tmp_path / "live.pcap"),
        "window_index": 0,
        "window_start": 0.0,
        "window_end": 5.0,
        "packet_count": 3,
        "image_path": str(output_path),
    }

    with patch("flowpic.capture.capture_packets_to_pcap") as capture_mock, patch(
        "flowpic.capture.generate_flowpic_preview", return_value=expected_row
    ) as preview_mock:
        row, pcap_path = capture_and_generate_flowpic(
            output_image=output_path,
            output_pcap=tmp_path / "live.pcap",
            duration=12,
            bpf_filter="tcp or udp",
        )

    capture_mock.assert_called_once()
    preview_mock.assert_called_once()
    assert row == expected_row
    assert pcap_path == tmp_path / "live.pcap"


def test_list_capture_interfaces_returns_sorted_names() -> None:
    with patch("flowpic.capture._load_capture_tools", return_value=(object(), object(), lambda: ["Wi-Fi", "Ethernet 2"])):
        interfaces = list_capture_interfaces()

    assert interfaces == ["Ethernet 2", "Wi-Fi"]


def test_build_dataset_writes_global_manifest(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    app_dir = raw_root / "YouTube"
    app_dir.mkdir(parents=True)
    (app_dir / "session1.pcap").write_bytes(b"pcap")

    fake_row = {
        "label": "YouTube",
        "source_pcap": str(app_dir / "session1.pcap"),
        "window_index": 0,
        "window_start": 0.0,
        "window_end": 5.0,
        "packet_count": 5,
        "image_path": str(processed_root / "YouTube" / "session1_w0000.png"),
    }

    with patch("flowpic.data_builder.process_pcap_to_flowpics", return_value=[fake_row]):
        rows = build_dataset(raw_root=raw_root, processed_root=processed_root)

    assert len(rows) == 1
    assert (processed_root / "manifest.csv").exists()


def test_train_model_saves_checkpoint(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest_path = tmp_path / "manifest.csv"

    samples = [
        ("AppA", image_dir / "a1.png", 0),
        ("AppA", image_dir / "a2.png", 16),
        ("AppB", image_dir / "b1.png", 255),
        ("AppB", image_dir / "b2.png", 220),
    ]

    for _, path, value in samples:
        Image.new("L", (32, 32), color=value).save(path)

    write_manifest(
        [
            {
                "label": label,
                "source_pcap": "synthetic.pcap",
                "window_index": index,
                "window_start": 0.0,
                "window_end": 5.0,
                "packet_count": 1,
                "image_path": str(path),
            }
            for index, (label, path, _) in enumerate(samples)
        ],
        manifest_path,
    )

    checkpoint_path = tmp_path / "backbone_v1.pth"
    result = train_model(manifest_path, checkpoint_path, epochs=1, batch_size=4)
    assert checkpoint_path.exists()
    assert result["num_classes"] == 2


def test_register_and_predict_pipeline(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    checkpoint_path = tmp_path / "model.pth"
    library_path = tmp_path / "library.json"

    model = FlowPicEmbeddingNet()
    torch.save({"model_state_dict": model.state_dict(), "embedding_dim": 128}, checkpoint_path)

    known_a = image_dir / "known_a.png"
    known_b = image_dir / "known_b.png"
    unknown = image_dir / "unknown.png"
    Image.new("L", (32, 32), color=32).save(known_a)
    Image.new("L", (32, 32), color=48).save(known_b)
    Image.new("L", (32, 32), color=32).save(unknown)

    template = register_application(
        label="YouTube",
        checkpoint_path=checkpoint_path,
        library_path=library_path,
        image_paths=[known_a, known_b],
    )
    assert template["num_samples"] == 2
    assert "YouTube" in load_template_library(library_path)["templates"]

    prediction = predict_input(
        checkpoint_path=checkpoint_path,
        library_path=library_path,
        image_path=unknown,
        threshold=0.5,
    )
    assert prediction["label"] == "YouTube"
    assert prediction["is_unknown"] is False


def test_demo_assets_and_run_pipeline(tmp_path: Path) -> None:
    assets = create_demo_assets(tmp_path / "demo")
    assert Path(assets["train_manifest"]).exists()
    assert Path(assets["zero_day_manifest"]).exists()

    result = run_demo(tmp_path / "demo_run", epochs=4, threshold=0.08)
    assert result["demo_claim"]["known_match"] is True
    assert result["demo_claim"]["zero_day_unknown_before"] is True
    assert result["demo_claim"]["zero_day_known_after"] is True
