from typing import Dict, Any, Optional, Tuple, List
import copy

def log(*args, **kwargs):
    pass

def deepcopy_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(scene)