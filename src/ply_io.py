"""
ply_io.py - High-Performance 3D Gaussian Splatting PLY Parser & Exporter

Provides pure NumPy/PyTorch binary little-endian and ASCII PLY parsing and serialization
specifically designed for 3D Gaussian Splatting representations.

Supports:
- Standard Inria / Graphdeco 3DGS PLY attributes:
  x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2,
  rot_0, rot_1, rot_2, rot_3 (with log-scale and logit-opacity conversions).
- Direct linear attributes:
  x, y, z, red, green, blue (or r, g, b), opacity, scale_0..2, rot_0..3.
- Robust header validation, endianness checks, NaN/Inf sanitization, and structured errors.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi)) for spherical harmonic DC to RGB


class PLYParseError(Exception):
    """Raised when a PLY file cannot be parsed or validated."""
    pass


def parse_ply_header(header_lines: List[str]) -> Tuple[int, str, List[Tuple[str, str]], int]:
    """
    Parses the ASCII header lines of a PLY file.
    
    Returns:
        num_vertices: Total vertex/Gaussian count.
        ply_format: Format string ('binary_little_endian 1.0', 'ascii 1.0', etc.)
        properties: List of (property_name, property_type) tuples.
        header_byte_length: Offset in bytes where header ends.
    """
    if not header_lines or header_lines[0].strip() != "ply":
        raise PLYParseError("Not a valid PLY file: missing initial 'ply' magic header.")

    ply_format = None
    num_vertices = None
    properties = []
    in_vertex_element = False

    for line in header_lines:
        tokens = line.strip().split()
        if not tokens:
            continue

        keyword = tokens[0]

        if keyword == "format":
            if len(tokens) < 3:
                raise PLYParseError(f"Malformed format line in PLY header: {line}")
            ply_format = f"{tokens[1]} {tokens[2]}"

        elif keyword == "element":
            if len(tokens) >= 3 and tokens[1] == "vertex":
                in_vertex_element = True
                try:
                    num_vertices = int(tokens[2])
                except ValueError:
                    raise PLYParseError(f"Invalid vertex count in PLY header: {tokens[2]}")
            else:
                in_vertex_element = False

        elif keyword == "property" and in_vertex_element:
            if len(tokens) >= 3:
                prop_type = tokens[1]
                prop_name = tokens[2]
                properties.append((prop_name, prop_type))

        elif keyword == "end_header":
            break

    if ply_format is None:
        raise PLYParseError("PLY header missing 'format' specification.")
    if num_vertices is None:
        raise PLYParseError("PLY header missing 'element vertex <count>' definition.")
    if not properties:
        raise PLYParseError("PLY header missing vertex property definitions.")

    return num_vertices, ply_format, properties


def _numpy_dtype_for_ply_type(ply_type: str) -> np.dtype:
    """Maps PLY property types to corresponding numpy dtypes."""
    mapping = {
        "char": np.int8,
        "int8": np.int8,
        "uchar": np.uint8,
        "uint8": np.uint8,
        "short": np.int16,
        "int16": np.int16,
        "ushort": np.uint16,
        "uint16": np.uint16,
        "int": np.int32,
        "int32": np.int32,
        "uint": np.uint32,
        "uint32": np.uint32,
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
        "float64": np.float64,
    }
    if ply_type not in mapping:
        raise PLYParseError(f"Unsupported PLY property type: '{ply_type}'")
    return mapping[ply_type]


def load_3dgs_ply(
    file_path_or_bytes: Union[str, Path, bytes],
    device: Optional[Union[str, torch.device]] = None,
    max_gaussians: int = 2_000_000,
) -> Dict[str, torch.Tensor]:
    """
    Loads a 3D Gaussian Splatting PLY file into PyTorch tensors.
    
    Returns a dictionary containing:
        means: [N, 3] float32 tensor of 3D positions (x, y, z)
        scales: [N, 3] float32 tensor of positive anisotropic scales
        quats: [N, 4] float32 tensor of unit quaternions [w, x, y, z]
        opacities: [N] float32 tensor of opacities in [0, 1]
        colors: [N, 3] float32 tensor of RGB colors in [0, 1]
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1. Read raw bytes
    if isinstance(file_path_or_bytes, (str, Path)):
        p = Path(file_path_or_bytes)
        if not p.exists():
            raise FileNotFoundError(f"PLY file does not exist: {p}")
        with open(p, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(file_path_or_bytes, bytes):
        raw_bytes = file_path_or_bytes
    else:
        raise TypeError("file_path_or_bytes must be a Path, path string, or bytes object.")

    if len(raw_bytes) < 20:
        raise PLYParseError("PLY file is too small or empty.")

    # 2. Extract ASCII header
    header_end_idx = raw_bytes.find(b"end_header\n")
    header_delimiter_len = len(b"end_header\n")
    if header_end_idx == -1:
        header_end_idx = raw_bytes.find(b"end_header\r\n")
        header_delimiter_len = len(b"end_header\r\n")
    if header_end_idx == -1:
        raise PLYParseError("Corrupt PLY file: 'end_header' tag not found.")

    header_text = raw_bytes[:header_end_idx].decode("ascii", errors="replace")
    header_lines = [line.strip() for line in header_text.splitlines() if line.strip()]
    data_start_offset = header_end_idx + header_delimiter_len

    num_vertices, ply_format, properties = parse_ply_header(header_lines)

    if num_vertices <= 0:
        raise PLYParseError(f"PLY file contains zero vertices ({num_vertices}).")
    if num_vertices > max_gaussians:
        raise PLYParseError(
            f"Scene contains {num_vertices:,} Gaussians, exceeding the safe demo limit of {max_gaussians:,}."
        )

    # 3. Construct structured numpy dtype
    dtype_fields = []
    for prop_name, prop_type in properties:
        np_t = _numpy_dtype_for_ply_type(prop_type)
        dtype_fields.append((prop_name, np_t))

    vertex_struct_dtype = np.dtype(dtype_fields)
    vertex_size_bytes = vertex_struct_dtype.itemsize

    # 4. Parse binary or ASCII data
    if "binary_little_endian" in ply_format:
        expected_bytes = num_vertices * vertex_size_bytes
        actual_bytes = len(raw_bytes) - data_start_offset
        if actual_bytes < expected_bytes:
            raise PLYParseError(
                f"Incomplete binary PLY file: expected {expected_bytes} bytes for {num_vertices} vertices, "
                f"but found only {actual_bytes} bytes after header."
            )
        data_buffer = np.frombuffer(
            raw_bytes, dtype=vertex_struct_dtype, count=num_vertices, offset=data_start_offset
        )

    elif "ascii" in ply_format:
        ascii_body = raw_bytes[data_start_offset:].decode("ascii", errors="replace")
        lines = [line.strip() for line in ascii_body.splitlines() if line.strip()]
        if len(lines) < num_vertices:
            raise PLYParseError(
                f"Incomplete ASCII PLY: expected {num_vertices} lines, found {len(lines)}."
            )
        # Parse ascii lines into structured array
        data_buffer = np.zeros(num_vertices, dtype=vertex_struct_dtype)
        for i in range(num_vertices):
            vals = lines[i].split()
            for j, (name, _) in enumerate(dtype_fields):
                data_buffer[name][i] = vals[j]

    else:
        raise PLYParseError(f"Unsupported PLY format: '{ply_format}'. Supported formats: binary_little_endian, ascii.")

    # 5. Extract 3DGS Components with flexible attribute naming
    prop_names = set(data_buffer.dtype.names)

    # Positions: x, y, z (Strictly Required)
    if not {"x", "y", "z"}.issubset(prop_names):
        raise PLYParseError(f"PLY file is missing required spatial attributes 'x', 'y', 'z'. Found: {list(prop_names)}")

    means_np = np.stack([data_buffer["x"], data_buffer["y"], data_buffer["z"]], axis=-1).astype(np.float32)
    means = torch.from_numpy(means_np).to(device)

    # Scales: scale_0, scale_1, scale_2
    if {"scale_0", "scale_1", "scale_2"}.issubset(prop_names):
        scales_raw = np.stack([data_buffer["scale_0"], data_buffer["scale_1"], data_buffer["scale_2"]], axis=-1).astype(np.float32)
        # Check if values are in log-space (standard Inria 3DGS stores log scales)
        # If scale has negative values, it is guaranteed log-scale
        if np.any(scales_raw < 0.0) or np.all(np.abs(scales_raw) < 15.0):
            scales_np = np.exp(np.clip(scales_raw, -12.0, 5.0))
        else:
            scales_np = np.clip(scales_raw, 1e-4, 10.0)
    else:
        # Fallback isotropic scales if omitted
        scales_np = np.ones((num_vertices, 3), dtype=np.float32) * 0.02

    scales = torch.from_numpy(scales_np).to(device)

    # Rotations: rot_0, rot_1, rot_2, rot_3 (quaternions [w, x, y, z] or [x, y, z, w])
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(prop_names):
        quats_raw = np.stack([data_buffer["rot_0"], data_buffer["rot_1"], data_buffer["rot_2"], data_buffer["rot_3"]], axis=-1).astype(np.float32)
        # Normalize quaternions
        norm = np.linalg.norm(quats_raw, axis=-1, keepdims=True)
        norm[norm < 1e-8] = 1.0
        quats_np = quats_raw / norm
    else:
        # Identity quaternion [1, 0, 0, 0]
        quats_np = np.zeros((num_vertices, 4), dtype=np.float32)
        quats_np[:, 0] = 1.0

    quats = torch.from_numpy(quats_np).to(device)

    # Opacity
    if "opacity" in prop_names:
        opac_raw = data_buffer["opacity"].astype(np.float32)
        # If opac_raw has values outside [0, 1] or negative, it is stored as logit
        if np.any(opac_raw < 0.0) or np.any(opac_raw > 1.0):
            opac_np = 1.0 / (1.0 + np.exp(-np.clip(opac_raw, -10.0, 10.0)))
        else:
            opac_np = np.clip(opac_raw, 0.0, 1.0)
    else:
        opac_np = np.ones(num_vertices, dtype=np.float32) * 0.85

    opacities = torch.from_numpy(opac_np).to(device)

    # Colors: f_dc_0, f_dc_1, f_dc_2 (SH DC) OR red, green, blue (uint8 or float)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(prop_names):
        sh_dc = np.stack([data_buffer["f_dc_0"], data_buffer["f_dc_1"], data_buffer["f_dc_2"]], axis=-1).astype(np.float32)
        # RGB = 0.5 + SH_C0 * f_dc
        colors_np = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
    elif {"red", "green", "blue"}.issubset(prop_names):
        rgb_raw = np.stack([data_buffer["red"], data_buffer["green"], data_buffer["blue"]], axis=-1).astype(np.float32)
        if np.any(rgb_raw > 1.0):
            colors_np = np.clip(rgb_raw / 255.0, 0.0, 1.0)
        else:
            colors_np = np.clip(rgb_raw, 0.0, 1.0)
    elif {"r", "g", "b"}.issubset(prop_names):
        rgb_raw = np.stack([data_buffer["r"], data_buffer["g"], data_buffer["b"]], axis=-1).astype(np.float32)
        if np.any(rgb_raw > 1.0):
            colors_np = np.clip(rgb_raw / 255.0, 0.0, 1.0)
        else:
            colors_np = np.clip(rgb_raw, 0.0, 1.0)
    else:
        # Default white/neutral color
        colors_np = np.ones((num_vertices, 3), dtype=np.float32) * 0.8

    colors = torch.from_numpy(colors_np).to(device)

    # Sanitize any NaNs or Infs
    means = torch.nan_to_num(means, nan=0.0, posinf=1.0, neginf=-1.0)
    scales = torch.nan_to_num(scales, nan=0.01, posinf=0.1, neginf=0.001).clamp(min=1e-4)
    quats = torch.nan_to_num(quats, nan=0.0, posinf=1.0, neginf=-1.0)
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
    opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
    }


def save_3dgs_ply(
    scene: Dict[str, torch.Tensor],
    out_path: Union[str, Path],
    as_inria_format: bool = True,
) -> Path:
    """
    Serializes a 3D Gaussian Splatting scene dictionary to standard binary little-endian PLY.
    
    If as_inria_format is True, writes log-scale, logit-opacity, and f_dc_0..2 properties
    matching standard 3DGS Inria conventions.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    means = scene["means"].detach().cpu().numpy().astype(np.float32)
    scales = scene["scales"].detach().cpu().numpy().astype(np.float32)
    quats = scene["quats"].detach().cpu().numpy().astype(np.float32)
    opacities = scene["opacities"].detach().cpu().numpy().astype(np.float32)
    colors = scene["colors"].detach().cpu().numpy().astype(np.float32)

    num_vertices = means.shape[0]

    if as_inria_format:
        # Convert to Inria 3DGS conventions: log scale, logit opacity, f_dc
        log_scales = np.log(np.clip(scales, 1e-6, 1e3))
        logit_opacities = np.log(np.clip(opacities / (1.0 - np.clip(opacities, 1e-5, 1.0 - 1e-5)), 1e-5, 1e5))
        f_dc = (colors - 0.5) / SH_C0

        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ])

        data = np.zeros(num_vertices, dtype=dtype)
        data["x"] = means[:, 0]
        data["y"] = means[:, 1]
        data["z"] = means[:, 2]
        data["f_dc_0"] = f_dc[:, 0]
        data["f_dc_1"] = f_dc[:, 1]
        data["f_dc_2"] = f_dc[:, 2]
        data["opacity"] = logit_opacities
        data["scale_0"] = log_scales[:, 0]
        data["scale_1"] = log_scales[:, 1]
        data["scale_2"] = log_scales[:, 2]
        data["rot_0"] = quats[:, 0]
        data["rot_1"] = quats[:, 1]
        data["rot_2"] = quats[:, 2]
        data["rot_3"] = quats[:, 3]

        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {num_vertices}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property float nx\n"
            f"property float ny\n"
            f"property float nz\n"
            f"property float f_dc_0\n"
            f"property float f_dc_1\n"
            f"property float f_dc_2\n"
            f"property float opacity\n"
            f"property float scale_0\n"
            f"property float scale_1\n"
            f"property float scale_2\n"
            f"property float rot_0\n"
            f"property float rot_1\n"
            f"property float rot_2\n"
            f"property float rot_3\n"
            f"end_header\n"
        )
    else:
        # Linear format
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ])
        data = np.zeros(num_vertices, dtype=dtype)
        data["x"] = means[:, 0]
        data["y"] = means[:, 1]
        data["z"] = means[:, 2]
        data["red"] = (colors[:, 0] * 255.0).astype(np.uint8)
        data["green"] = (colors[:, 1] * 255.0).astype(np.uint8)
        data["blue"] = (colors[:, 2] * 255.0).astype(np.uint8)
        data["opacity"] = opacities
        data["scale_0"] = scales[:, 0]
        data["scale_1"] = scales[:, 1]
        data["scale_2"] = scales[:, 2]
        data["rot_0"] = quats[:, 0]
        data["rot_1"] = quats[:, 1]
        data["rot_2"] = quats[:, 2]
        data["rot_3"] = quats[:, 3]

        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {num_vertices}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property uchar red\n"
            f"property uchar green\n"
            f"property uchar blue\n"
            f"property float opacity\n"
            f"property float scale_0\n"
            f"property float scale_1\n"
            f"property float scale_2\n"
            f"property float rot_0\n"
            f"property float rot_1\n"
            f"property float rot_2\n"
            f"property float rot_3\n"
            f"end_header\n"
        )

    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    return out_path
