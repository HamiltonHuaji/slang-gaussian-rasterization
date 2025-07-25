  # Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import slangpy
import os
from os import PathLike
from pathlib import Path

shaders_path = Path(__file__).parent.absolute()
device = slangpy.create_device(include_paths=[shaders_path], enable_cuda_interop=True, enable_print=True)

TILE_SIZES_HW = [(4,4), (8,8), (16,16)]

vertex_shader = slangpy.TorchModule.load_from_file(device, "vertex_shader.slang")
tile_shader = slangpy.TorchModule.load_from_file(device, "tile_shader.slang")

with open(shaders_path / "alphablend_shader.slang", "r") as f:
    alpha_blend_shader_source = f.read()

alpha_blend_shaders = {}
for tile_height, tile_width in TILE_SIZES_HW:
    alpha_blend_shaders[(tile_height, tile_width)] = slangpy.TorchModule.load_from_source(
        device, "alphablend_shader.slang",
        alpha_blend_shader_source.replace('PYTHON_TILE_HEIGHT', str(tile_height)).replace('PYTHON_TILE_WIDTH', str(tile_width)),
    )
