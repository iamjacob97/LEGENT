from typing import Dict, Any
import copy

from legent.scene_generation.types import Vector3

def log(*args, **kwargs):
    pass

def deepcopy_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(scene)

def convert_vector(position_or_rotation: Vector3):
    x = position_or_rotation["x"]
    y = position_or_rotation["y"]
    z = position_or_rotation["z"]
    return (x, y, z)
