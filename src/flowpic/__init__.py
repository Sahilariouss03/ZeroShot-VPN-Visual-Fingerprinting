"""FlowPic VPN fingerprinting package."""

__all__ = [
    "FlowPicEmbeddingNet",
    "build_dataset",
    "capture_and_generate_flowpic",
    "generate_flowpic_preview",
    "list_capture_interfaces",
    "match_embedding",
    "predict_input",
    "register_application",
    "train_model",
]


def __getattr__(name: str):
    if name == "build_dataset":
        from .data_builder import build_dataset

        return build_dataset
    if name == "capture_and_generate_flowpic":
        from .capture import capture_and_generate_flowpic

        return capture_and_generate_flowpic
    if name == "generate_flowpic_preview":
        from .generate import generate_flowpic_preview

        return generate_flowpic_preview
    if name == "list_capture_interfaces":
        from .capture import list_capture_interfaces

        return list_capture_interfaces
    if name == "match_embedding":
        from .matching import match_embedding

        return match_embedding
    if name == "FlowPicEmbeddingNet":
        from .model import FlowPicEmbeddingNet

        return FlowPicEmbeddingNet
    if name == "predict_input":
        from .predict import predict_input

        return predict_input
    if name == "register_application":
        from .library import register_application

        return register_application
    if name == "train_model":
        from .train_pipeline import train_model

        return train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
