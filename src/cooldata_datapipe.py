# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
This is the datapipe to read OpenFoam files (vtp/vtu/stl) and save them as point clouds
in npy format.

"""

from abc import abstractmethod
import time, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, TypedDict, Union, Callable

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from physicsnemo.utils.domino.utils import *
from torch.utils.data import Dataset
from cooldata.pyvista_flow_field_dataset import PyvistaFlowFieldDataset

# AIR_DENSITY = 1.205
# STREAM_VELOCITY = 30.00

class Normalization:

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, x: np.ndarray) -> np.ndarray:
        pass

class ZScoreNormalization(Normalization):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

class ScaleNormalization(Normalization):
    def __init__(self, scale: float):
        self.scale = scale

    def encode(self, x: np.ndarray) -> np.ndarray:
        return x / self.scale

    def decode(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale


class CoolDataset(Dataset):
    """
    Datapipe for converting openfoam dataset to npy

    """

    def __init__(
        self,
        data_path: Union[str, Path],
        out_data_path: Union[str, Path],
        surface_variables: Optional[list] = [
            "Pressure",
            "Temperature",
            "HeatTransferCoefficient",
            "WallShearStress_0",
            "WallShearStress_1",
            "WallShearStress_2",
        ],
        volume_variables: Optional[list] = ["Temperature", "Pressure", "Velocity_0", "Velocity_1", "Velocity_2"],
        normalizations: dict[str, Normalization] = {
            "Pressure": ScaleNormalization(3.0),
            "Temperature": ZScoreNormalization(295.0, 5.0),
            "Velocity_0": ZScoreNormalization(3.5, 2.0),
            "Velocity_1": ScaleNormalization(0.2),
            "Velocity_2": ScaleNormalization(0.1),
            "WallShearStress_0": ScaleNormalization(1.0),
            "WallShearStress_1": ScaleNormalization(0.1),
            "WallShearStress_2": ScaleNormalization(0.1),
            "HeatTransferCoefficient": ScaleNormalization(400.0),
        },
        device: int = 0,
        model_type=None,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()

        self.data_path = data_path
        self.out_data_path = Path(out_data_path).expanduser()


        assert self.data_path.exists(), f"Path {self.data_path} does not exist"

        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"
        self.pv_ds = PyvistaFlowFieldDataset.load_from_huggingface(self.data_path,num_samples=3)
        self.pv_ds.shuffle()

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables

        # self.stream_velocity = 0.0
        # for vel_component in self.global_params_reference["inlet_velocity"]:
        #     self.stream_velocity += vel_component**2
        # self.stream_velocity = np.sqrt(self.stream_velocity)
        # self.air_density = self.global_params_reference["air_density"]
        self.normalizations = normalizations

        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.pv_ds)

    def __getitem__(self, idx):
        for component in self.pv_ds[idx].surface_data[0]:
            if "WallShearStress_0" not in component.cell_data.keys():
                component.cell_data["WallShearStress_0"] = np.zeros(component.n_cells)
                component.cell_data["WallShearStress_1"] = np.zeros(component.n_cells)
                component.cell_data["WallShearStress_2"] = np.zeros(component.n_cells)
        surface_pv: pv.UnstructuredGrid = self.pv_ds[idx].surface_data.combine().extract_surface().triangulate()
        stl_vertices = surface_pv.points
        stl_faces = np.array(surface_pv.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = surface_pv.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(surface_pv.cell_centers().points)


        if self.model_type == "volume" or self.model_type == "combined":

            # Get the unstructured grid data
            polydata = self.pv_ds[idx].volume_data[0][0][0].cell_data_to_point_data()
            volume_coordinates, volume_fields = get_volume_data(
                polydata, self.volume_variables
            )
            volume_fields = np.concatenate(volume_fields, axis=-1)

            # Non-dimensionalize volume fields
            for i, var in enumerate(self.volume_variables):
                if var in self.normalizations:
                    volume_fields[:, i] = self.normalizations[var].encode(
                        volume_fields[:, i]
                    )
                else:
                    raise ValueError(f"Normalization for {var} not provided")
        else:
            volume_fields = None
            volume_coordinates = None

        if self.model_type == "surface" or self.model_type == "combined":
            polydata = surface_pv

            celldata_all = get_node_to_elem(polydata)
            celldata = celldata_all.GetCellData()
            surface_fields = get_fields(celldata, self.surface_variables)
            surface_fields = np.concatenate(surface_fields, axis=-1)

            mesh = pv.PolyData(polydata)
            surface_coordinates = np.array(mesh.cell_centers().points)

            surface_normals = np.array(mesh.cell_normals)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"])

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )

            # Non-dimensionalize surface fields
            for i, var in enumerate(self.surface_variables):
                if var in self.normalizations:
                    surface_fields[:, i] = self.normalizations[var].encode(
                        surface_fields[:, i]
                    )
                else:
                    raise ValueError(f"Normalization for {var} not provided")
        else:
            surface_fields = None
            surface_coordinates = None
            surface_normals = None
            surface_sizes = None

        # Arrange global parameters reference in a list based on the type of the parameter
        global_params_reference_list = []
        # for name, type in self.global_params_types.items():
        #     if type == "vector":
        #         global_params_reference_list.extend(self.global_params_reference[name])
        #     elif type == "scalar":
        #         global_params_reference_list.append(self.global_params_reference[name])
        #     else:
        #         raise ValueError(
        #             f"Global parameter {name} not supported for  this dataset"
        #         )
        # global_params_reference = np.array(
        #     global_params_reference_list, dtype=np.float32
        # )
        global_params_reference = None

        # Prepare the list of global parameter values for each simulation file
        # Note: The user must ensure that the values provided here correspond to the
        # `global_parameters` specified in `config.yaml` and that these parameters
        # exist within each simulation file.
        global_params_values_list = []
        for key in sorted(self.normalizations.keys()):
            val = self.normalizations[key]
            if isinstance(val, ScaleNormalization):
                global_params_values_list.append(val.scale)
            elif isinstance(val, ZScoreNormalization):
                global_params_values_list.append(val.mean)
                global_params_values_list.append(val.std)
            else:
                raise ValueError(f"Normalization for {key} not supported")
        global_params_values = np.array(global_params_values_list, dtype=np.float32)

        # Add the parameters to the dictionary
        return {
            "stl_coordinates": np.float32(stl_vertices),
            "stl_centers": np.float32(stl_centers),
            "stl_faces": np.float32(mesh_indices_flattened),
            "stl_areas": np.float32(stl_sizes),
            "surface_mesh_centers": np.float32(surface_coordinates),
            "surface_normals": np.float32(surface_normals),
            "surface_areas": np.float32(surface_sizes),
            "volume_fields": np.float32(volume_fields),
            "volume_mesh_centers": np.float32(volume_coordinates),
            "surface_fields": np.float32(surface_fields),
            "filename": self.pv_ds[idx].volume_path,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
        }


if __name__ == "__main__":
    fm_data = OpenFoamDataset(
        data_path="/code/aerofoundationdata/",
        phase="train",
        volume_variables=["UMean", "pMean", "nutMean"],
        surface_variables=["pMean", "wallShearStress", "nutMean"],
        global_params_types={"inlet_velocity": "vector", "air_density": "scalar"},
        global_params_reference={"inlet_velocity": [30.0], "air_density": 1.226},
        sampling=False,
        sample_in_bbox=False,
    )
    d_dict = fm_data[1]
