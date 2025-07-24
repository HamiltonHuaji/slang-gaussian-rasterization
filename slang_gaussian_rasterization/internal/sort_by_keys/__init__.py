import torch
import torch.utils.cpp_extension
from pathlib import Path

root_path = Path(__file__).parent

sort_by_keys_cub = torch.utils.cpp_extension.load(
    name="sort_by_keys", 
    sources=[str(root_path / "sort_by_keys.cu")]
)
