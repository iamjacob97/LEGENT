from typing import Dict, Any, Optional, Tuple, List
import os
import trimesh
import importlib.util

from hazalyser.utils import convert_vector
from legent.environment.env import Environment, Action 

package_name = "legent"
package_path = importlib.util.find_spec(package_name).submodule_search_locations[0]

# Construct the path to the resource
resource_path = os.path.join(package_path, os.pardir, "hazalyser")
resource_path = os.path.abspath(resource_path)

AGENT_PATH = os.path.join(resource_path, "obsAssets", "agent")
SUBJECT_PATH = os.path.join(resource_path, "obsAssets", "subject")
LLM_ANALYSIS_PATH = os.path.join(resource_path, "llm_analysis")

class SceneBundle:
    def __init__(self, scene_generator, infos, max_floor_objects, 
    spawnable_asset_groups, spawnable_assets, specified_object_types, 
    max_object_types_per_room, hidden_object_indices = None):
        self.generator = scene_generator
        self.infos = infos
        self.max_floor_objects = max_floor_objects
        self.spawnable_asset_groups = spawnable_asset_groups
        self.spawnable_assets = spawnable_assets
        self.specified_object_types = specified_object_types
        self.max_object_types_per_room = max_object_types_per_room
        self._hidden_object_indices = hidden_object_indices or []

def get_mesh_size(input_file):
    """Get the bounding box of a mesh file.
    Args:
        input_file: str, the path of the mesh file.
    Returns:
        mesh_size: np.ndarray, the size of the mesh file.
    """
    mesh = trimesh.load(input_file)
    min_vals, max_vals = mesh.bounds[0], mesh.bounds[1]
    return max_vals - min_vals
    
def get_current_scene_state(env: Environment, action: Action = Action()) -> Dict[str, Any]:
    obs = env.step(action)
    return obs.game_states

def update_position_and_rotation(scene: Dict[str, Any], game_states: Dict[str, Any]) -> None:
    keys = {"instances", "player", "agent"}
    convertibles = {"position", "rotation"}
    for key in keys:
        if key == "instances":
            scene_instances = scene[key]
            obs_instances = game_states[key]
            for obj_id, instance in enumerate(scene_instances):
                if instance["prefab"] in obs_instances[obj_id]["prefab"]:
                    for convertible in convertibles:
                        instance[convertible] = convert_vector(obs_instances[obj_id][convertible])
        else:
            for convertible in convertibles:
                scene[key][convertible] = convert_vector(game_states[key][convertible])                   

def is_structural(odb, inst: Dict[str, Any]) -> bool:
    #inefficient, for single instance checks
    structural_assets = odb.MY_OBJECTS
    wall_prefabs = set(structural_assets.get("wall", []))
    floor_prefabs = set(structural_assets.get("floor", []))
    walls_and_floors = wall_prefabs | floor_prefabs

    return inst["prefab"] in walls_and_floors

def is_small_object_instance(inst: Dict[str, Any]) -> bool:
    # Heuristic: small objects are interactables whose parent is a receptacle or have parent not -1
    return inst["type"] == "interactable"
        
def is_subject(inst: Dict[str, Any], subject: str) -> bool:
    return inst["prefab"] == os.path.join(SUBJECT_PATH, subject)

def matches(odb, inst: Dict[str, Any], name: str) -> bool:
    key = name.lower()
    if inst["prefab"].lower() == key:
        return True
    otype = odb.OBJECT_TO_TYPE.get(inst["prefab"], "").lower()
    return otype == key

