import io
import torch
import base64


def serialize_tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    1. Save tensor to an in-memory buffer using torch.save().
    2. Base64-encode that buffer's bytes to produce a string.
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    raw_bytes = buffer.read()
    b64_str = base64.b64encode(raw_bytes).decode('utf-8')
    return b64_str

def deserialize_tensor_from_base64(b64_str: str) -> torch.Tensor:
    """
    1. Convert base64-encoded string back to raw bytes.
    2. Load it using torch.load() to get a PyTorch tensor (or any object).
    """
    raw_bytes = base64.b64decode(b64_str)
    buffer = io.BytesIO(raw_bytes)
    tensor = torch.load(buffer)
    return tensor