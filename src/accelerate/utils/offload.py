# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import zfpy
import einops

from ..logging import get_logger
from .imports import is_safetensors_available


logger = get_logger(__name__)
compressed = dict()


# PCA: https://github.com/gngdb/pytorch-pca/blob/61b14afbb0c401e4c4992199497520262be10b14/pca.py#L60
def svd_flip_v(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    v *= signs.view(-1, 1)
    return v


class GpuPcaOffload:
    def __init__(self, weight: torch.Tensor):
        self.block_len = 32
        self.orig_feature_size = self.block_len * self.block_len
        self.compression_rate = 2
        self.n_components = int(self.block_len * self.block_len / self.compression_rate)
        self.orig_shape = weight.size()

        with torch.no_grad():
            weight = weight.to('cpu')
            orig_weight = weight
            if self.orig_shape[0] % self.block_len != 0 or self.orig_shape[1] % self.block_len != 0:
                weight = torch.nn.functional.pad(
                    weight,
                    (0, self.block_len - self.orig_shape[1], 0, self.block_len - self.orig_shape[0]),
                    'constant', 0)
            self.padded_shape = weight.size()
            weight = weight.unfold(0, self.block_len, self.block_len).unfold(1, self.block_len, self.block_len)
            weight = weight.reshape(
                (self.padded_shape[0]*self.padded_shape[1]) // self.orig_feature_size,
                self.orig_feature_size)
            self.mean = weight.mean(dim=(-2,), keepdim=True)
            weight.sub_(self.mean)
            U, S, V = torch.pca_lowrank(weight, q=self.n_components, center=False)
            self.components = svd_flip_v(U, V.T)
            self.projection = torch.matmul(weight, self.components.T)

            # Sanity check
            decompressed = self.decompress()
            decompressed = decompressed.to('cpu')
            good = torch.allclose(decompressed, orig_weight, atol=1e-6, rtol=1e-4)
            abs_dev = torch.max(torch.abs(orig_weight-decompressed))
            rel_dev = torch.max(torch.abs(orig_weight-decompressed)
                                / torch.maximum(torch.abs(orig_weight), torch.abs(decompressed)))

            print(f'All close: {good}, abs_dev={abs_dev} rel_dev={rel_dev}')

    # def decompress(self, device=None):
    #     if device is not None:
    #         mean = self.mean.to(device)
    #         components = self.components.to(device)
    #         projection = self.projection.to(device)
    #     approx = torch.matmul(projection, components)
    #     del components
    #     del projection
    #     approx.add_(mean)
    #     del mean
    #     # ans = einops.rearrange('(s1 s2) (f1 f2) -> (s1 f1) (s2 f2)')
    #     # https://stackoverflow.com/a/66784823/1915854
    #     tile_x, tile_y  = self.block_len, self.block_len
    #     n_x_splits = self.padded_shape[0] // self.block_len
    #     n_y_splits = self.padded_shape[1] // self.block_len
    #     approx = approx.reshape(n_x_splits, n_y_splits, tile_x, tile_y)
    #     approx = approx.permute(0, 2, 1, 3).reshape(tile_x * n_x_splits, tile_y * n_y_splits)
    #     ans = approx[:self.orig_shape[0], :self.orig_shape[1]]
    #     del approx
    #     gc.collect()
    #     return ans

    def decompress(self, device=None):
        if device is not None:
            self.mean = self.mean.to(device)
            self.components = self.components.to(device)
            self.projection = self.projection.to(device)
        approx = torch.matmul(self.projection, self.components)
        approx.add_(self.mean)
        # ans = einops.rearrange('(s1 s2) (f1 f2) -> (s1 f1) (s2 f2)')
        # https://stackoverflow.com/a/66784823/1915854
        tile_x, tile_y  = self.block_len, self.block_len
        n_x_splits = self.padded_shape[0] // self.block_len
        n_y_splits = self.padded_shape[1] // self.block_len
        approx = approx.reshape(n_x_splits, n_y_splits, tile_x, tile_y)
        approx = approx.permute(0, 2, 1, 3).reshape(tile_x * n_x_splits, tile_y * n_y_splits)
        ans = approx[:self.orig_shape[0], :self.orig_shape[1]]
        return ans

class CpuZfpOffload:
    def __init__(self, weight: np.ndarray):
        self.comp_data = None
        self.orig_data = None
        # if np.issubdtype(weight.dtype, np.floating):
        #     self.comp_data = zfpy.compress_numpy(weight, rate=16)  # 16 bit per floating point number
        # else:
        self.orig_data = weight

    def decompress(self) -> np.ndarray:
        # if self.comp_data is not None:
        #     return zfpy.decompress_numpy(self.comp_data)
        # else:
        return self.orig_data


def move_offloads(offload_folder, prefix, offload_index):
    extension = ".dat"
    new_compressed = dict()
    folder_str = str(offload_folder)
    if not folder_str.endswith('/'):
        folder_str += '/'
    global compressed
    for key, val in compressed.items():
        if not key.startswith(folder_str):
            new_compressed[key] = val
            continue
        if not key.endswith(extension):
            new_compressed[key] = val
            continue
        weight_name = key[len(folder_str):-len(extension)]
        if weight_name not in offload_index:
            new_compressed[key] = val
            continue
        new_key = f"{folder_str}{prefix}.{weight_name}{extension}"
        new_compressed[new_key] = val
    compressed = new_compressed


def offload_weight(weight, weight_name, offload_folder, index=None):
    tensor_file = os.path.join(offload_folder, f"{weight_name}.dat")
    if weight.dim() == 2 and torch.is_floating_point(weight):
        compressed[tensor_file] = GpuPcaOffload(weight)
        if index is not None:
            dtype = str(weight.dtype)
            torch_prefix = 'torch.'
            if dtype.startswith(torch_prefix):
                dtype = dtype[len(torch_prefix):]
            index[weight_name] = {"dtype": dtype, "shape": list(weight.size())}
    else:
        dtype = None
        # Check the string instead of the dtype to be compatible with versions of PyTorch that don't have bfloat16.
        if str(weight.dtype) == "torch.bfloat16":
            # Need to reinterpret the underlined data as int16 since NumPy does not handle bfloat16s.
            weight = weight.view(torch.int16)
            dtype = "bfloat16"
        array = weight.cpu().numpy()
        if index is not None:
            if dtype is None:
                dtype = str(array.dtype)
            index[weight_name] = {"dtype": dtype, "shape": list(array.shape)}
        if array.ndim == 0:
            array = array[None]
        compressed[tensor_file] = CpuZfpOffload(array)
        # file_array = np.memmap(tensor_file, dtype=array.dtype, mode="w+", shape=array.shape)
        # file_array[:] = array[:]
        # file_array.flush()
    return index


def load_offloaded_weight(weight_file, weight_info, device=None):
    comp_obj = compressed[weight_file]
    if isinstance(comp_obj, CpuZfpOffload):
        shape = tuple(weight_info["shape"])
        if shape == ():
            # NumPy memory-mapped arrays can't have 0 dims so it was saved as 1d tensor
            shape = (1,)

        dtype = weight_info["dtype"]
        if dtype == "bfloat16":
            # NumPy does not support bfloat16 so this was saved as a int16
            dtype = "int16"

        # weight = np.memmap(weight_file, dtype=dtype, shape=shape, mode="r")
        weight = comp_obj.decompress()

        if len(weight_info["shape"]) == 0:
            weight = weight[0]
        weight = torch.tensor(weight)
        if weight_info["dtype"] == "bfloat16":
            weight = weight.view(torch.bfloat16)
    elif isinstance(comp_obj, GpuPcaOffload):
        weight = comp_obj.decompress(device)
    else:
        raise RuntimeError(f"Unhandled compressed object type: {str(comp_obj)}")
    return weight


def save_offload_index(index, offload_folder):
    if index is None or len(index) == 0:
        # Nothing to save
        return

    offload_index_file = os.path.join(offload_folder, "index.json")
    if os.path.isfile(offload_index_file):
        with open(offload_index_file, "r", encoding="utf-8") as f:
            current_index = json.load(f)
    else:
        current_index = {}
    current_index.update(index)

    with open(offload_index_file, "w", encoding="utf-8") as f:
        json.dump(current_index, f, indent=2)


def offload_state_dict(save_dir: Union[str, os.PathLike], state_dict: Dict[str, torch.Tensor]):
    """
    Offload a state dict in a given folder.

    Args:
        save_dir (`str` or `os.PathLike`):
            The directory in which to offload the state dict.
        state_dict (`Dict[str, torch.Tensor]`):
            The dictionary of tensors to offload.
    """
    os.makedirs(save_dir, exist_ok=True)
    index = {}
    for name, parameter in state_dict.items():
        index = offload_weight(parameter, name, save_dir, index=index)

    # Update index
    save_offload_index(index, save_dir)


class PrefixedDataset(Mapping):
    """
    Will access keys in a given dataset by adding a prefix.

    Args:
        dataset (`Mapping`): Any map with string keys.
        prefix (`str`): A prefix to add when trying to access any element in the underlying dataset.
    """

    def __init__(self, dataset: Mapping, prefix: str):
        self.dataset = dataset
        self.prefix = prefix

    def __getitem__(self, key):
        return self.dataset[f"{self.prefix}{key}"]

    def __iter__(self):
        return iter([key for key in self.dataset if key.startswith(self.prefix)])

    def __len__(self):
        return len(self.dataset)


class OffloadedWeightsLoader(Mapping):
    """
    A collection that loads weights stored in a given state dict or memory-mapped on disk.

    Args:
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            A dictionary parameter name to tensor.
        save_folder (`str` or `os.PathLike`, *optional*):
            The directory in which the weights are stored (by `offload_state_dict` for instance).
        index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype`/ `shape` or safetensors filename). Will default
            to the index saved in `save_folder`.
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor] = None,
        save_folder: Optional[Union[str, os.PathLike]] = None,
        index: Mapping = None,
        device=None,
    ):
        if state_dict is None and save_folder is None:
            raise ValueError("Need either a `state_dict` or a `save_folder` containing offloaded weights.")

        self.state_dict = {} if state_dict is None else state_dict
        self.save_folder = save_folder
        if index is None and save_folder is not None:
            with open(os.path.join(save_folder, "index.json")) as f:
                index = json.load(f)
        self.index = {} if index is None else index
        self.all_keys = list(self.state_dict.keys())
        self.all_keys.extend([key for key in self.index if key not in self.all_keys])
        self.device = device

    def __getitem__(self, key: str):
        # State dict gets priority
        if key in self.state_dict:
            return self.state_dict[key]
        weight_info = self.index[key]
        if weight_info.get("safetensors_file") is not None:
            if not is_safetensors_available():
                raise ImportError("These offloaded weights require the use of safetensors: `pip install safetensors`.")

            if "SAFETENSORS_FAST_GPU" not in os.environ:
                logger.info("Enabling fast loading with safetensors by setting `SAFETENSORS_FAST_GPU` to 1.")
                os.environ["SAFETENSORS_FAST_GPU"] = "1"

            from safetensors import safe_open

            device = "cpu" if self.device is None else self.device
            with safe_open(weight_info["safetensors_file"], framework="pt", device=device) as f:
                tensor = f.get_tensor(weight_info.get("weight_name", key))

            if "dtype" in weight_info:
                return tensor.to(getattr(torch, weight_info["dtype"]))
            else:
                return tensor

        weight_file = os.path.join(self.save_folder, f"{key}.dat")
        return load_offloaded_weight(weight_file, weight_info, device=self.device)

    def __iter__(self):
        return iter(self.all_keys)

    def __len__(self):
        return len(self.all_keys)


def extract_submodules_state_dict(state_dict: Dict[str, torch.Tensor], submodule_names: List[str]):
    """
    Extract the sub state-dict corresponding to a list of given submodules.

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dict to extract from.
        submodule_names (`List[str]`): The list of submodule names we want to extract.
    """
    result = {}
    for module_name in submodule_names:
        # We want to catch module_name parameter (module_name.xxx) or potentially module_name, but not any of the
        # submodules that could being like module_name (transformers.h.1 and transformers.h.10 for instance)
        result.update(
            {
                key: param
                for key, param in state_dict.items()
                if key == module_name or key.startswith(module_name + ".")
            }
        )
    return result
