from __future__ import annotations

import argparse
from pathlib import Path

from .generate import FlowPicManifestRow, generate_flowpic_preview


def _load_capture_tools():
    try:
        from scapy.all import get_if_list, sniff, wrpcap
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Scapy is required for live capture. Install project dependencies first."
        ) from exc

    return sniff, wrpcap, get_if_list


def list_capture_interfaces() -> list[dict[str, str]]:
    _, _, get_if_list = _load_capture_tools()
    ifaces = get_if_list()

    mapping = {}
    if hasattr(get_if_list, "__module__") and "scapy" in get_if_list.__module__:
        try:
            from scapy.arch.windows import get_windows_if_list
            for win_iface in get_windows_if_list():
                if "guid" in win_iface:
                    npf_name = f"\\Device\\NPF_{win_iface['guid']}"
                    friendly = win_iface.get("name") or win_iface.get("description") or win_iface["guid"]
                    desc = win_iface.get("description", "")
                    if friendly != desc and desc:
                        friendly = f"{friendly} ({desc})"
                    mapping[npf_name] = friendly
                    # Windows loopback is sometimes just NPF_Loopback
                    if win_iface.get("name") == "Software Loopback Interface 1":
                        mapping["\\Device\\NPF_Loopback"] = "Local Loopback"
        except ImportError:
            pass

    results = []
    for iface in ifaces:
        if_str = str(iface)
        # Fallback for standard loopback if missing from registry
        if if_str == "\\Device\\NPF_Loopback" and if_str not in mapping:
            mapping[if_str] = "Local Loopback"
        results.append({
            "id": if_str,
            "name": mapping.get(if_str, if_str),
        })

    return sorted(results, key=lambda x: x["name"])


def capture_packets_to_pcap(
    output_pcap: str | Path,
    *,
    duration: int = 10,
    packet_count: int = 0,
    interface: str | None = None,
    bpf_filter: str | None = None,
):
    sniff, wrpcap, _ = _load_capture_tools()
    output_file = Path(output_pcap)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sniff_kwargs = {"timeout": duration, "store": True}
    if packet_count > 0:
        sniff_kwargs["count"] = packet_count
    if interface:
        sniff_kwargs["iface"] = interface
    if bpf_filter:
        sniff_kwargs["filter"] = bpf_filter

    packets = sniff(**sniff_kwargs)
    if len(packets) == 0:
        raise ValueError(
            "Live capture finished without IP traffic. Try generating traffic, changing the interface, or increasing duration."
        )

    wrpcap(str(output_file), packets)
    return packets


def capture_and_generate_flowpic(
    *,
    output_image: str | Path,
    output_pcap: str | Path | None = None,
    duration: int = 10,
    packet_count: int = 0,
    interface: str | None = None,
    bpf_filter: str | None = None,
    label: str = "live_capture",
    window_seconds: float = 5.0,
    bins_time: int = 96,
    bins_size: int = 96,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
    show: bool = False,
    manifest_path: str | Path | None = None,
    window_index: int = -1,
) -> tuple[FlowPicManifestRow, Path]:
    image_path = Path(output_image)
    pcap_path = Path(output_pcap) if output_pcap else image_path.with_suffix(".pcap")

    capture_packets_to_pcap(
        pcap_path,
        duration=duration,
        packet_count=packet_count,
        interface=interface,
        bpf_filter=bpf_filter,
    )
    row = generate_flowpic_preview(
        pcap_path=pcap_path,
        output_path=image_path,
        label=label,
        window_seconds=window_seconds,
        bins_time=bins_time,
        bins_size=bins_size,
        time_feature=time_feature,
        max_packet_size=max_packet_size,
        show=show,
        manifest_path=manifest_path,
        window_index=window_index,
    )
    return row, pcap_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture live traffic and generate a FlowPic.")
    parser.add_argument("--output", help="Path to the output PNG file.")
    parser.add_argument("--pcap-output", help="Optional path for the saved PCAP capture.")
    parser.add_argument("--duration", type=int, default=10, help="Capture duration in seconds.")
    parser.add_argument("--packet-count", type=int, default=0, help="Optional early stop after N packets.")
    parser.add_argument("--interface", help="Interface name passed to Scapy sniff().")
    parser.add_argument("--filter", dest="bpf_filter", help="Optional BPF capture filter, for example 'tcp or udp'.")
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="Print available Scapy capture interfaces and exit.",
    )
    parser.add_argument("--label", default="live_capture", help="Label to store in the manifest row.")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window size in seconds.")
    parser.add_argument("--window-index", type=int, default=-1, help="Which non-empty window to render. (-1 for auto)")
    parser.add_argument("--bins-time", type=int, default=96, help="Number of histogram bins on the x-axis.")
    parser.add_argument("--bins-size", type=int, default=96, help="Number of histogram bins on the y-axis.")
    parser.add_argument(
        "--time-feature",
        choices=["arrival_time", "inter_arrival"],
        default="arrival_time",
        help="Time feature to map on the x-axis.",
    )
    parser.add_argument("--max-packet-size", type=int, default=1600, help="Maximum packet size used for clipping.")
    parser.add_argument("--manifest", help="Optional manifest CSV output path.")
    parser.add_argument("--show", action="store_true", help="Display the FlowPic interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_interfaces:
        for interface in list_capture_interfaces():
            print(f"{interface['name']}  [{interface['id']}]")
        return

    if not args.output:
        raise SystemExit("--output is required unless --list-interfaces is used.")

    capture_and_generate_flowpic(
        output_image=args.output,
        output_pcap=args.pcap_output,
        duration=args.duration,
        packet_count=args.packet_count,
        interface=args.interface,
        bpf_filter=args.bpf_filter,
        label=args.label,
        window_seconds=args.window_seconds,
        bins_time=args.bins_time,
        bins_size=args.bins_size,
        time_feature=args.time_feature,
        max_packet_size=args.max_packet_size,
        show=args.show,
        manifest_path=args.manifest,
        window_index=args.window_index,
    )


if __name__ == "__main__":
    main()
