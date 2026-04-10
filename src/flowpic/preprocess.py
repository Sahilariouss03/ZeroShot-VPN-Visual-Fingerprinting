from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PacketRecord:
    timestamp: float
    relative_time: float
    inter_arrival_time: float
    packet_size: int
    direction: int


@dataclass(frozen=True)
class FlowWindow:
    index: int
    start_time: float
    end_time: float
    actual_end_time: float
    packets: list[PacketRecord]


def _load_scapy():
    try:
        from scapy.all import IP, IPv6, PcapReader, TCP, UDP
    except ImportError as exc:  # pragma: no cover - import path varies by environment
        raise RuntimeError(
            "Scapy is required for PCAP processing. Install project dependencies first."
        ) from exc

    return PcapReader, IP, IPv6, TCP, UDP


def packet_record_from_scapy_packet(
    packet,
    *,
    first_timestamp: float | None,
    previous_timestamp: float | None,
) -> PacketRecord | None:
    PcapReader, IP, IPv6, TCP, UDP = _load_scapy()
    if not (packet.haslayer(IP) or packet.haslayer(IPv6)):
        return None

    timestamp = float(packet.time)
    base_timestamp = timestamp if first_timestamp is None else first_timestamp
    relative_time = timestamp - base_timestamp
    inter_arrival_time = 0.0 if previous_timestamp is None else max(0.0, timestamp - previous_timestamp)

    ip_layer = packet.getlayer(IP) or packet.getlayer(IPv6)
    direction = 1
    if packet.haslayer(TCP):
        tcp_layer = packet.getlayer(TCP)
        direction = 1 if int(tcp_layer.sport) <= int(tcp_layer.dport) else -1
    elif packet.haslayer(UDP):
        udp_layer = packet.getlayer(UDP)
        direction = 1 if int(udp_layer.sport) <= int(udp_layer.dport) else -1
    elif ip_layer is not None:
        direction = 1 if str(getattr(ip_layer, "src", "")) <= str(getattr(ip_layer, "dst", "")) else -1

    return PacketRecord(
        timestamp=timestamp,
        relative_time=relative_time,
        inter_arrival_time=inter_arrival_time,
        packet_size=len(packet),
        direction=direction,
    )


def iter_packet_records(pcap_path: str | Path) -> Iterable[PacketRecord]:
    pcap_file = Path(pcap_path)
    PcapReader, _, _, _, _ = _load_scapy()

    first_timestamp: float | None = None
    previous_timestamp: float | None = None

    with PcapReader(str(pcap_file)) as packets:
        for packet in packets:
            record = packet_record_from_scapy_packet(
                packet,
                first_timestamp=first_timestamp,
                previous_timestamp=previous_timestamp,
            )
            if record is None:
                continue
            if first_timestamp is None:
                first_timestamp = record.timestamp
            previous_timestamp = record.timestamp
            yield record


def window_packet_records(
    packet_records: Iterable[PacketRecord],
    window_seconds: float = 5.0,
    allow_single_short_window: bool = True,
) -> list[FlowWindow]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    records = list(packet_records)
    if not records:
        return []

    windows: dict[int, list[PacketRecord]] = {}
    for record in records:
        window_index = int(record.relative_time // window_seconds)
        windows.setdefault(window_index, []).append(record)

    flow_windows: list[FlowWindow] = []
    for window_index in sorted(windows):
        packets = windows[window_index]
        start_time = window_index * window_seconds
        end_time = start_time + window_seconds
        flow_windows.append(
            FlowWindow(
                index=window_index,
                start_time=start_time,
                end_time=end_time,
                actual_end_time=packets[-1].relative_time,
                packets=packets,
            )
        )

    if len(flow_windows) == 1 and allow_single_short_window:
        return flow_windows

    trailing_index = flow_windows[-1].index
    filtered_windows = [
        window
        for window in flow_windows
        if window.index != trailing_index or window.actual_end_time >= window.end_time
    ]
    return filtered_windows or flow_windows[:1]
