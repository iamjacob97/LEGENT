from typing import Dict, Any, Optional, Tuple, List
import os
import copy
import math
import random
import trimesh

from hazalyser.utils import convert_vector
from legent.environment.env import Environment, Action 
from legent.server.rect_placer import RectPlacer

#-----------controller helpers------------#
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

#------------generator helpers------------#

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

def get_asset_path(asset_folder: str="obsAssets") -> str:
    """Get the path of a asset."""
    path = os.path.abspath(asset_folder)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Asset folder {path} does not exist.")

    return path










# def is_small_object_instance(self, inst: Dict[str, Any]) -> bool:
#     # Heuristic: small objects are interactables whose parent is a receptacle or have parent not -1
#     if inst.get("type") != "interactable":
#         return False
#     # Many small objects have parent set to receptacle prefab name, or room_id present with is_receptacle False
#     return inst.get("parent", -1) != -1 or not inst.get("is_receptacle", False)

# def matches(self, inst: Dict[str, Any], name: str) -> bool:
#     key = name.lower()
#     if inst["prefab"].lower() == key:
#         return True
#     otype = self.odb.OBJECT_TO_TYPE.get(inst["prefab"], "").lower()
#     return otype == key

# def scene_bbox(self, scene: Dict[str, Any]) -> Tuple[float, float, float, float]:
#     # Conservative bbox from room polygon if available
#     rx = [p for p in scene.get("room_polygon", [])]
#     if rx:
#         xs = []
#         zs = []
#         for r in rx:
#             for x, z in r.get("polygon", []):
#                 xs.append(x)
#                 zs.append(z)
#         if xs and zs:
#             return (min(xs), min(zs), max(xs), max(zs))
#     # Fallback from instances
#     xs = [inst["position"][0] for inst in scene["instances"]]
#     zs = [inst["position"][2] for inst in scene["instances"]]
#     return (min(xs) - 2.0, min(zs) - 2.0, max(xs) + 2.0, max(zs) + 2.0)

# def add_random_small_objects(
#     self,
#     scene: Dict[str, Any],
#     add_count: int,
#     allowed_room_types: Optional[set] = None,
# ) -> Dict[str, Any]:
#     # Build receptacle surfaces from current scene
#     placer = RectPlacer(self._scene_bbox(scene))
#     instances = list(scene["instances"])  # copy

#     receptacles = [
#         inst for inst in instances
#         if self.odb.OBJECT_TO_TYPE.get(inst["prefab"], None) in self.odb.RECEPTACLES
#     ]
#     surfaces = []
#     for rec in receptacles:
#         rec_prefab = rec["prefab"]
#         rec_rot = rec["rotation"][1] if isinstance(rec.get("rotation"), list) else 0
#         for surf in self.odb.PREFABS[rec_prefab].get("placeable_surfaces", []) or []:
#             surfaces.append((rec, rec_rot, surf))

#     # Sample suitable small object types by room weight
#     small_types = [t for t in self.odb.PLACEMENT_ANNOTATIONS.index if self.odb.PLACEMENT_ANNOTATIONS.loc[t]["onSurface"]]
#     weights = []
#     for t in small_types:
#         if allowed_room_types:
#             w = max(self.odb.PLACEMENT_ANNOTATIONS.loc[t][f"in{rt}s"] for rt in allowed_room_types)
#         else:
#             w = sum(self.odb.PLACEMENT_ANNOTATIONS.loc[t][f"in{rt}s"] for rt in ["Bedroom", "LivingRoom", "Kitchen", "Bathroom"]) / 4.0
#         weights.append(w)

#     for _ in range(add_count):
#         # pick type by weight then pick prefab by type
#         if not small_types:
#             break
#         t = random.choices(small_types, weights=weights, k=1)[0]
#         prefab = random.choice(self.odb.OBJECT_DICT[t])
#         placed = self._place_one_small_on_any_surface(instances, placer, prefab, surfaces)
#         if not placed:
#             # give up quietly
#             pass

#     scene2 = self._deepcopy_scene(scene)
#     scene2["instances"] = instances
#     return scene2

# def place_small_objects_by_type_or_prefab(self, scene: Dict[str, Any], name: str, count: int) -> int:
#     placer = RectPlacer(self._scene_bbox(scene))
#     instances = list(scene["instances"])  # copy

#     receptacles = [
#         inst for inst in instances
#         if self.odb.OBJECT_TO_TYPE.get(inst["prefab"], None) in self.odb.RECEPTACLES
#     ]
#     surfaces = []
#     for rec in receptacles:
#         rec_prefab = rec["prefab"]
#         rec_rot = rec["rotation"][1] if isinstance(rec.get("rotation"), list) else 0
#         for surf in self.odb.PREFABS[rec_prefab].get("placeable_surfaces", []) or []:
#             surfaces.append((rec, rec_rot, surf))

#     placed = 0
#     key = name.lower()
#     for _ in range(count):
#         if key in self.odb.OBJECT_DICT:
#             prefab = random.choice(self.odb.OBJECT_DICT[key])
#         elif name in self.odb.PREFABS:
#             prefab = name
#         else:
#             break
#         if self._place_one_small_on_any_surface(instances, placer, prefab, surfaces):
#             placed += 1

#     scene["instances"] = instances
#     return placed

# def place_one_small_on_any_surface(
#     self,
#     instances: List[Dict[str, Any]],
#     placer: RectPlacer,
#     prefab: str,
#     surfaces: List[Tuple[Dict[str, Any], float, Dict[str, float]]],
# ) -> bool:
#     random.shuffle(surfaces)
#     psize = self.odb.PREFABS[prefab]["size"]

#     for rec, rec_rot, surf in surfaces:
#         # For axis-aligned approximation, swap extents if rotated 90/270
#         x_min = (surf["x_min"] + rec["position"][0]) if rec_rot in (0, 180) else (surf["z_min"] + rec["position"][0])
#         x_max = (surf["x_max"] + rec["position"][0]) if rec_rot in (0, 180) else (surf["z_max"] + rec["position"][0])
#         z_min = (surf["z_min"] + rec["position"][2]) if rec_rot in (0, 180) else (surf["x_min"] + rec["position"][2])
#         z_max = (surf["z_max"] + rec["position"][2]) if rec_rot in (0, 180) else (surf["x_max"] + rec["position"][2])

#         x_margin = psize["x"] / 2 + 0.1
#         z_margin = psize["z"] / 2 + 0.1
#         sample_x_min = x_min + x_margin
#         sample_x_max = x_max - x_margin
#         sample_z_min = z_min + z_margin
#         sample_z_max = z_max - z_margin
#         if sample_x_min >= sample_x_max or sample_z_min >= sample_z_max:
#             continue

#         for _ in range(10):
#             x = random.uniform(sample_x_min, sample_x_max)
#             z = random.uniform(sample_z_min, sample_z_max)
#             sx = psize["x"] + 0.2
#             sz = psize["z"] + 0.2
#             if placer.place(prefab, x, z, sx, sz):
#                 y = rec["position"][1] + surf["y"] + psize["y"] / 2
#                 instances.append(
#                     {
#                         "prefab": prefab,
#                         "position": (x, y, z),
#                         "rotation": [0, 0, 0],
#                         "scale": [1, 1, 1],
#                         "type": "interactable",
#                         "parent": rec["prefab"],
#                     }
#                 )
#                 return True
#     return False

# def place_floor_receptacles_by_type(self, scene: Dict[str, Any], type_key: str, count: int) -> Dict[str, Any]:
#     # Place simple floor objects inside room bbox using RectPlacer, without anchor heuristics.
#     instances = list(scene["instances"])  # copy
#     placer = RectPlacer(self._scene_bbox(scene))
#     prefabs = self.odb.OBJECT_DICT.get(type_key, [])
#     for _ in range(count):
#         if not prefabs:
#             break
#         prefab = random.choice(prefabs)
#         psize = self.odb.PREFABS[prefab]["size"]
#         # Try a few random positions
#         placed = False
#         for _ in range(50):
#             # sample from bbox
#             x0, z0, x1, z1 = self._scene_bbox(scene)
#             x = random.uniform(x0 + 0.5, x1 - 0.5)
#             z = random.uniform(z0 + 0.5, z1 - 0.5)
#             sx = psize["x"]
#             sz = psize["z"]
#             # Inflate a bit to avoid near-walls
#             if placer.place(prefab, x, z, sx, sz):
#                 instances.append(
#                     {
#                         "prefab": prefab,
#                         "position": (x, psize["y"] / 2, z),
#                         "rotation": [0, random.choice([0, 90, 180, 270]), 0],
#                         "scale": [1, 1, 1],
#                         "type": (
#                             "interactable"
#                             if prefab in self.odb.KINETIC_AND_INTERACTABLE_INFO["interactable_names"]
#                             else "kinematic"
#                         ),
#                         "is_receptacle": True,
#                     }
#                 )
#                 placed = True
#                 break
#         if not placed:
#             break
#     scene2 = self._deepcopy_scene(scene)
#     scene2["instances"] = instances
#     return scene2

# def recluster_smalls_near(self, scene: Dict[str, Any], target_xz: Tuple[float, float]) -> Dict[str, Any]:
#     instances = []
#     smalls = []
#     for inst in scene["instances"]:
#         if self._is_small_object_instance(inst):
#             smalls.append(inst)
#         else:
#             instances.append(inst)

#     placer = RectPlacer(self._scene_bbox(scene))
#     tx, tz = target_xz
#     angle = random.uniform(0, 360)
#     radius = 0.5
#     for s in smalls:
#         psize = self.odb.PREFABS[s["prefab"]]["size"]
#         for _ in range(20):
#             angle += random.uniform(20, 90)
#             radius += 0.05
#             x = tx + radius * math.cos(math.radians(angle))
#             z = tz + radius * math.sin(math.radians(angle))
#             if placer.place(s["prefab"], x, z, psize["x"] + 0.1, psize["z"] + 0.1):
#                 s = dict(s)
#                 s["position"] = (x, s["position"][1], z)
#                 instances.append(s)
#                 break
#     scene2 = self._deepcopy_scene(scene)
#     scene2["instances"] = instances
#     return scene2

# def cluster_by_category(self, scene: Dict[str, Any]) -> Dict[str, Any]:
#     instances = []
#     smalls_by_type: Dict[str, List[Dict[str, Any]]] = {}
#     for inst in scene["instances"]:
#         if self._is_small_object_instance(inst):
#             t = self.odb.OBJECT_TO_TYPE.get(inst["prefab"], "unknown")
#             smalls_by_type.setdefault(t, []).append(inst)
#         else:
#             instances.append(inst)

#     bbox = self._scene_bbox(scene)
#     x0, z0, x1, z1 = bbox
#     xm = (x0 + x1) / 2
#     zm = (z0 + z1) / 2
#     quadrants = [
#         (x0, z0, xm, zm),
#         (xm, z0, x1, zm),
#         (x0, zm, xm, z1),
#         (xm, zm, x1, z1),
#     ]
#     q_index = 0
#     for _, items in smalls_by_type.items():
#         q = quadrants[q_index % len(quadrants)]
#         q_index += 1
#         placer = RectPlacer(q)
#         for s in items:
#             psize = self.odb.PREFABS[s["prefab"]]["size"]
#             for _ in range(30):
#                 x = random.uniform(q[0] + 0.2, q[2] - 0.2)
#                 z = random.uniform(q[1] + 0.2, q[3] - 0.2)
#                 if placer.place(s["prefab"], x, z, psize["x"] + 0.1, psize["z"] + 0.1):
#                     s = dict(s)
#                     s["position"] = (x, s["position"][1], z)
#                     instances.append(s)
#                     break
#     scene2 = self._deepcopy_scene(scene)
#     scene2["instances"] = instances
#     return scene2

# ---------- Chat-controlled workflow ----------

# # --- Chat-controlled workflow ---
# def configure_generation(self, room_type: RoomType, items: Optional[Dict[str, int]] = None, include_other_items: bool = True, dims: Optional[Tuple[int, int]] = None) -> None:
#     """Configure parameters used by reset_scene() and chat control."""
#     self._gen_params = {
#         "room_type": room_type,
#         "items": items,
#         "include_other_items": include_other_items,
#         "dims": dims,
#     }

# # ---------- Controls after selection ----------
# def level_of_clutter(self, value: float) -> None:
#     """
#     Adjust the number of small objects in the locked scene.
#     - value in [0.0, 1.0]; scales towards sparser (0) or denser (1).

#     Strategy: sample additional small objects from room-appropriate types and
#     remove some if needed, while respecting collision via RectPlacer.
#     """
#     value = max(0.0, min(1.0, float(value)))

#     scene = self._get_active_scene(copy_scene=True)
#     # Current small objects (interactable and parent != -1 indicates placed on surfaces)
#     smalls_idx = [i for i, inst in enumerate(scene["instances"]) if self._is_small_object_instance(inst)]

#     # Target scale around baseline
#     baseline = len(smalls_idx)
#     target = int(max(0, math.floor(baseline * (0.25 + 1.5 * value))))

#     if target < baseline:
#         # Remove some small objects at random
#         to_remove = set(random.sample(smalls_idx, baseline - target)) if baseline - target > 0 else set()
#         scene["instances"] = [inst for idx, inst in enumerate(scene["instances"]) if idx not in to_remove]
#     elif target > baseline:
#         # Add more small objects of room-appropriate types
#         room_types = set(p["room_type"] for p in scene.get("room_polygon", []))
#         add_count = target - baseline
#         scene = self._add_random_small_objects(scene, add_count, allowed_room_types=room_types)

#     self._apply_scene(scene)

# def spacing(self, value: float) -> None:
#     """
#     Enforce a minimum spacing between small objects by pruning conflicts.
#     - value in [0.0, 1.0]; converts to a world-space margin.
#     """
#     value = max(0.0, min(1.0, float(value)))
#     min_margin = 0.05 + 0.45 * value  # meters

#     scene = self._get_active_scene(copy_scene=True)
#     placer = RectPlacer(self._scene_bbox(scene))
#     kept: List[Dict[str, Any]] = []

#     for inst in scene["instances"]:
#         if not self._is_small_object_instance(inst):
#             kept.append(inst)
#             continue
#         px, pz = inst["position"][0], inst["position"][2]
#         sx = self.odb.PREFABS[inst["prefab"]]["size"]["x"] + 2 * min_margin
#         sz = self.odb.PREFABS[inst["prefab"]]["size"]["z"] + 2 * min_margin
#         if placer.place(inst["prefab"], px, pz, sx, sz):
#             kept.append(inst)

#     scene["instances"] = kept
#     self._apply_scene(scene)

# def arrange_by(self, parameter: str, reference: Optional[str] = None) -> None:
#     """
#     Basic arrangements:
#     - parameter == "proximity": cluster small objects near a reference prefab/type name
#     - parameter == "category": cluster by object types in quadrants
#     """
#     scene = self._get_active_scene(copy_scene=True)

#     if parameter == "proximity" and reference:
#         ref_positions = [inst["position"] for inst in scene["instances"] if self._matches(inst, reference)]
#         if ref_positions:
#             ref_x, _, ref_z = ref_positions[0]
#             scene = self._recluster_smalls_near(scene, (ref_x, ref_z))
#     elif parameter == "category":
#         scene = self._cluster_by_category(scene)

#     self._apply_scene(scene)

# def remove_item(self, item_name: str, count: int = 1) -> None:
#     scene = self._get_active_scene(copy_scene=True)
#     remaining = count
#     new_instances: List[Dict[str, Any]] = []
#     for inst in scene["instances"]:
#         if remaining > 0 and self._matches(inst, item_name):
#             remaining -= 1
#             continue
#         new_instances.append(inst)
#     scene["instances"] = new_instances
#     self._apply_scene(scene)

# def add_item(self, item_name: str, count: int = 1) -> None:
#     """
#     Add small objects (type or prefab) by placing them on available surfaces.
#     If a receptacle/floor asset type is given, attempt to add the receptacle as a floor object.
#     """
#     scene = self._get_active_scene(copy_scene=True)

#     # Try small object path first
#     placed = self._place_small_objects_by_type_or_prefab(scene, item_name, count)
#     if placed < count and item_name.lower() in self.odb.OBJECT_DICT:  # fallback/use remaining as floor asset type
#         rest = count - placed
#         scene = self._place_floor_receptacles_by_type(scene, item_name.lower(), rest)

#     self._apply_scene(scene)

# # ---------- External asset registration ----------
# def register_external_asset(
#     self,
#     asset_path: str,
#     asset_type: str,
#     *,
#     inBedrooms: int = 2,
#     inLivingRooms: int = 2,
#     inKitchens: int = 2,
#     inBathrooms: int = 2,
#     onFloor: bool = True,
#     multiplePerRoom: bool = True,
# ) -> None:
#     """
#     Register a custom 3D asset so it can be referenced in the items dict and used by the procedural algorithm.

#     - asset_path: path to a 3D file (e.g., .glb/.gltf/.obj). Used as the prefab name.
#     - asset_type: semantic type bucket (lowercase), e.g., "toolbox" or "medicalkit".
#     - room weights and placement flags control spawn rules similar to ProcTHOR annotations.
#     """
#     import os
#     import json
#     import pandas as pd
#     from pathlib import Path
#     from legent.environment.env_utils import get_default_env_data_path
#     import importlib

#     data_root = Path(f"{get_default_env_data_path()}/procthor")
#     assert os.path.exists(asset_path), f"Asset not found: {asset_path}"

#     # 1) Compute mesh size and add to addressables.json
#     addr_path = data_root / "addressables.json"
#     addressables = json.load(open(addr_path, "r", encoding="utf-8"))
#     try:
#         trimesh = importlib.import_module("trimesh")
#     except Exception as exc:
#         raise RuntimeError("trimesh is required to register external assets. Please install it (pip install trimesh).") from exc
#     mesh = trimesh.load(asset_path)
#     min_vals, max_vals = mesh.bounds[0], mesh.bounds[1]
#     size = (max_vals - min_vals).tolist()
#     prefab_entry = {
#         "name": asset_path,
#         "size": {"x": size[0], "y": size[1], "z": size[2]},
#         "placeable_surfaces": [
#             {"y": size[1] / 2, "x_min": -size[0] / 2, "z_min": -size[2] / 2, "x_max": size[0] / 2, "z_max": size[2] / 2}
#         ],
#         "type": "kinematic",
#     }
#     if prefab_entry not in addressables["prefabs"]:
#         addressables["prefabs"].append(prefab_entry)
#         json.dump(addressables, open(addr_path, "w", encoding="utf-8"), indent=4)

#     # 2) Ensure asset_type exists in object_dict.json
#     objdict_path = data_root / "object_dict.json"
#     object_dict = json.load(open(objdict_path, "r", encoding="utf-8"))
#     asset_type_l = asset_type.lower()
#     if asset_type_l not in object_dict:
#         object_dict[asset_type_l] = []
#     if asset_path not in object_dict[asset_type_l]:
#         object_dict[asset_type_l].append(asset_path)
#         json.dump(object_dict, open(objdict_path, "w", encoding="utf-8"), indent=4)

#     # 3) Map prefab name to type in object_name_to_type.json
#     n2t_path = data_root / "object_name_to_type.json"
#     name2type = json.load(open(n2t_path, "r", encoding="utf-8"))
#     if asset_path not in name2type:
#         name2type[asset_path] = asset_type_l
#         json.dump(name2type, open(n2t_path, "w", encoding="utf-8"), indent=4)

#     # 4) If new type, add placement annotations and receptacle permissions
#     pa_path = data_root / "placement_annotations.csv"
#     df = pd.read_csv(pa_path)
#     if asset_type_l not in set(df["assetType"]) and asset_type_l not in set(df.get("Object", [])):
#         # Normalize column names: LEGENT sheet may use "assetType" or "Object" as index depending on version
#         col = "assetType" if "assetType" in df.columns else ("Object" if "Object" in df.columns else None)
#         if col is None:
#             raise RuntimeError("placement_annotations.csv missing required identifier column")
#         new_row = {
#             col: asset_type_l,
#             "inKitchens": inKitchens,
#             "inLivingRooms": inLivingRooms,
#             "inBedrooms": inBedrooms,
#             "inBathrooms": inBathrooms,
#             "onFloor": onFloor,
#             "multiplePerRoom": multiplePerRoom,
#             "onEdge": True,
#             "inMiddle": True,
#             "inCorner": True,
#         }
#         df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#         df.to_csv(pa_path, index=False)

#         # Receptacle spawn permissions: allow small objects of this type to appear in all receptacles by default
#         rec_path = data_root / "receptacle.json"
#         receptacles = json.load(open(rec_path, "r", encoding="utf-8"))
#         for k in receptacles:
#             receptacles[k][asset_type_l] = 2
#         json.dump(receptacles, open(rec_path, "w", encoding="utf-8"), indent=4)

#     # 5) Refresh object DB so the new asset is visible immediately
#     try:
#         # Reset the cached default DB in LEGENT
#         from legent.scene_generation import objects as _objmod
#         _objmod.DEFAULT_OBJECT_DB = None
#     except Exception:
#         pass
#     self.odb = get_default_object_db()
