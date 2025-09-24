import os
import json
import copy
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from hazalyser.helpers import SUBJECT_PATH, get_mesh_size, convert_vector, log
from hazalyser.house import HazardRoom, HazardRoomSpec, HazardHouse, generate_house_structure
from hazalyser.objects import ObjectDB, get_default_object_db
from hazalyser.smallObjects import add_small_objects
from legent.scene_generation.generator import HouseGenerator
from legent.server.rect_placer import RectPlacer
from legent.scene_generation.room import Room
from legent.scene_generation.asset_groups import Asset
from legent.utils.math import look_rotation

RoomType = str  # "Bedroom" | "LivingRoom" | "Kitchen" | "Bathroom"

@dataclass
class SceneConfig:
    room_spec: HazardRoomSpec = HazardRoomSpec(room_spec_id="ESHA-SingleRoom", 
                                               spec=HazardRoom(room_id=1, 
                                               room_type=random.choice(["Bedroom", "LivingRoom", "Kitchen", "Bathroom"])))
    dims: Optional[Tuple[int, int]] = None # (x_size, z_size) in cell units
    agent: Optional[str] = None
    agent_scale: Optional[Tuple] = (1, 1, 1)
    agent_info: Optional[str] = ""
    subject: Optional[str] = None 
    subject_scale: Optional[Tuple] = (1, 1, 1)
    subject_info: Optional[str] = ""
    task: Optional[str] = ""
    items: Optional[Dict[str, int]] = None  # user-specified dictionary; keys are LEGENT types (e.g., "orange", "table")
    framework: Optional[str] = ""
    method: Optional[str] = ""
    llm_key: Optional[str] = ""
    vision_support: bool = False
    temperature: int = None
    max_tokens: int = None
    analysis_folder: str = "analysis"

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
        self.clutter_level = 0
        self.spacing_level = 0

DEFAULT_UNIT_SIZE = 2.5
DEFAULT_FLOOR_SIZE = 2.5
MAX_PLACEMENT_ATTEMPTS = 9
WALL_THICKNESS = 0.075

class SceneGenerator(HouseGenerator):
    def __init__(self, scene_config: SceneConfig = SceneConfig(), odb: ObjectDB = get_default_object_db(), unit_size = DEFAULT_UNIT_SIZE):
        self.scene_config = scene_config
        self.odb = odb
        self.rooms: Dict[str, Room] = dict()
        self.unit_size = unit_size
        self.half_unit_size = unit_size / 2  # Half of the size of a unit in the grid
        self.scale_ratio = unit_size / DEFAULT_FLOOR_SIZE

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"scene_config={self.scene_config!r}, "
                f"odb={type(self.odb).__name__}, "
                f"rooms={len(self.rooms)}, "
                f"unit_size={self.unit_size}, "
                f"scale_ratio={self.scale_ratio:.2f})") 
    
    def generate_structure(self, dims=None) -> HazardHouse:
        house_structure = generate_house_structure(room_spec=self.scene_config.room_spec, dims=dims or self.scene_config.dims, unit_size=self.unit_size)
        return house_structure
    
    def add_floors_and_walls(self, house_structure, room_spec, odb, prefabs, wall = None, floor = None, add_ceiling = False, remove_out_walls = False):
        """single room with no doors."""
        room_id = room_spec.spec.room_id
        wall_prefab = wall or np.random.choice(odb.MY_OBJECTS["wall"])
        room2wall = {room_id: wall_prefab, 0: wall_prefab}
        
        WALL_PREFAB = room2wall[room_id]
        wall_x_size, wall_y_size, wall_z_size = (
            prefabs[WALL_PREFAB]["size"]["x"],
            prefabs[WALL_PREFAB]["size"]["y"],
            prefabs[WALL_PREFAB]["size"]["z"],
        )
        log(
            f"wall_x_size: {wall_x_size}, wall_y_size: {wall_y_size}, wall_z_size: {wall_z_size}"
        )
        floor_prefab = floor or np.random.choice(odb.MY_OBJECTS["floor"])
        room2floor = {room_id: floor_prefab}
        FLOOR_PREFAB = room2floor[room_id]
        floor_x_size, floor_y_size, floor_z_size = (
            prefabs[FLOOR_PREFAB]["size"]["x"],
            prefabs[FLOOR_PREFAB]["size"]["y"],
            prefabs[FLOOR_PREFAB]["size"]["z"],
        )
        log(
            f"floor_x_size: {floor_x_size}, floor_y_size: {floor_y_size}, floor_z_size: {floor_z_size}"
        )
        floors = house_structure.floorplan

        floor_instances = []
        # generate walls based on the 0-1 boundaries
        for i in range(floors.shape[0]):
            for j in range(floors.shape[1]):
                if floors[i][j] != 0:
                    FLOOR_PREFAB = room2floor[floors[i][j]]

                    x, z = (i + 0.5 - 1) * self.unit_size, (j + 0.5 - 1) * self.unit_size
                    floor_instances.append(
                        {
                            "prefab": FLOOR_PREFAB,
                            "position": [x, -floor_y_size / 2, z],
                            "rotation": [0, 90, 0],
                            "scale": [self.scale_ratio, 1, self.scale_ratio],
                            "type": "kinematic",
                        }
                    )
                    # add ceiling
                    if add_ceiling:
                        floor_instances.append(
                            {
                                "prefab": FLOOR_PREFAB,
                                "position": [x, wall_y_size + floor_y_size / 2, z],
                                "rotation": [0, 90, 0],
                                "scale": [self.scale_ratio, 1, self.scale_ratio],
                                "type": "kinematic",
                            }
                        )

                WALL_PREFAB = room2wall[floors[i][j]]
                wall_x_size, wall_y_size, wall_z_size = (
                    prefabs[WALL_PREFAB]["size"]["x"],
                    prefabs[WALL_PREFAB]["size"]["y"],
                    prefabs[WALL_PREFAB]["size"]["z"],
                )
                
                a = floors[i][j]
                if i < floors.shape[0] - 1:
                    a_col = floors[i + 1][j]
                    if remove_out_walls and (a==0 or a_col==0):
                        continue

                    if a != a_col:
                        x = i + 1 - 1
                        z = j + 0.5 - 1

                        x = x * self.unit_size
                        z = z * self.unit_size

                        left_x = x - wall_z_size / 4
                        right_x = x + wall_z_size / 4

                        left_wall_prefab = room2wall[floors[i][j]]
                        right_wall_prefab = room2wall[floors[i + 1][j]]

                        left_scale = [
                            self.scale_ratio,
                            self.align_wall_height_scale(left_wall_prefab),
                            0.5,
                        ]
                        right_scale = [
                            self.scale_ratio,
                            self.align_wall_height_scale(right_wall_prefab),
                            0.5,
                        ]


                        left_rotation = 270
                        right_rotation = 90
                        
                        left_wall = self.format_object(
                            left_wall_prefab,
                            (left_x, 1.5, z),
                            left_rotation,
                            left_scale,
                        )
                        right_wall = self.format_object(
                            right_wall_prefab,
                            (right_x, 1.5, z),
                            right_rotation,
                            right_scale,
                        )

                        if floors[i][j] != 0:
                            floor_instances.append(left_wall)
                        if floors[i + 1][j] != 0:
                            floor_instances.append(right_wall)

                if j < floors.shape[1] - 1:
                    a_row = floors[i][j + 1]
                    if remove_out_walls and (a==0 or a_row==0):
                        continue

                    if a != a_row:
                        x = i + 0.5 - 1
                        z = j + 1 - 1

                        x = x * self.unit_size
                        z = z * self.unit_size

                        up_z = z - wall_z_size / 4
                        down_z = z + wall_z_size / 4

                        up_wall_prefab = room2wall[floors[i][j]]
                        down_wall_prefab = room2wall[floors[i][j + 1]]

                        up_scale = [
                            self.scale_ratio,
                            self.align_wall_height_scale(up_wall_prefab),
                            0.5,
                        ]
                        down_scale = [
                            self.scale_ratio,
                            self.align_wall_height_scale(down_wall_prefab),
                            0.5,
                        ]

                        up_rotation = 180
                        down_rotation = 0

                        up_wall = self.format_object(
                                up_wall_prefab,
                                (x, 1.5, up_z),
                                up_rotation,
                                up_scale,
                        )
                        down_wall = self.format_object(
                            down_wall_prefab,
                            (x, 1.5, down_z),
                            down_rotation,
                            down_scale,
                        )
                        if floors[i][j] != 0:
                            floor_instances.append(up_wall)
                        if floors[i][j + 1] != 0:
                            floor_instances.append(down_wall)
                        
        return floor_instances, floors

    def get_rooms(self, room_spec: HazardRoomSpec, floor_polygons) -> None:
        room_id = room_spec.spec.room_id
        room_type = room_spec.spec.room_type
        polygon = floor_polygons[f"room|{room_id}"]
        room = Room(polygon=polygon, room_type=room_type, room_id=room_id, odb=self.odb)
        self.rooms[room_id] = room

    def add_player_agent_subject(self, floors):
        def get_bbox_of_floor(x, z):
            x, z = (x - 0.5) * self.unit_size, (z - 0.5) * self.unit_size
            return (
                x - self.half_unit_size,
                z - self.half_unit_size,
                x + self.half_unit_size,
                z + self.half_unit_size,
            )

        def random_xz_for_agent(eps, floors):  # To prevent being positioned in the wall and getting pushed out by collision detection.
            # ravel the floor
            ravel_floors = floors.ravel()
            # get the index of the floor
            floor_idx = np.where(ravel_floors != 0)[0]
            # sample from the floor index
            floor_idx = np.random.choice(floor_idx)
            # get the x and z index
            x, z = np.unravel_index(floor_idx, floors.shape)
            log(f"human/agent x: {x}, z: {z}")

            # get the bbox of the floor
            bbox = get_bbox_of_floor(x, z)
            # uniformly sample from the bbox, with eps
            x, z = np.random.uniform(bbox[0] + eps, bbox[2] - eps), np.random.uniform(bbox[1] + eps, bbox[3] - eps)
            return x, z

        ### STEP 3: Randomly place the player and playmate (AI agent)
        # place the player
        AGENT_HUMAN_SIZE = 1
        while True:
            x, z = random_xz_for_agent(eps=0.5, floors=floors)
            player = {
                "prefab": "",
                "position": [x, 0.05, z],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": [1, 1, 1],
                "parent": -1,
                "type": "",
            }
            ok = self.placer.place("player", x, z, AGENT_HUMAN_SIZE, AGENT_HUMAN_SIZE)

            if ok:
                log(f"player x: {x}, z: {z}")
                break
        # place the playmate
        while True:
            x, z = random_xz_for_agent(eps=0.5, floors=floors)
            playmate = {
                "prefab": "",
                "position": [x, 0.05, z],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": [1, 1, 1],
                "parent": -1,
                "type": "",
            }
            ok = self.placer.place("playmate", x, z, AGENT_HUMAN_SIZE, AGENT_HUMAN_SIZE)
            if ok:
                log(f"playmate x: {x}, z: {z}")
                break
        #place the subject
        subject = []
        if self.scene_config.subject:
            asset = self.scene_config.subject
            asset_path = os.path.join(SUBJECT_PATH, asset)
            assert os.path.exists(asset_path), f"{asset_path} does not exist"
            subject_size = get_mesh_size(asset_path)
            subject_scale = self.scene_config.subject_scale

            x_size = subject_size[0]
            z_size = subject_size[2]
            while True:
                x, z = random_xz_for_agent(eps=0.5, floors=floors)
                
                ok = self.placer.place("subject", x, z, x_size, z_size)
                if ok:
                    subject.append({
                        "prefab": asset_path,
                        "position": [x, (subject_size[1] * subject_scale[1])/2, z],
                        "rotation": [0, np.random.uniform(0, 360), 0],
                        "scale": subject_scale,
                        "parent": -1,
                        "type": "kinematic",
                    })
                    log(f"subject x: {x}, z: {z}")
                    break

        # player lookat the playmate
        vs, vt = np.array(player["position"]), np.array(playmate["position"])
        vr = look_rotation(vt - vs)
        player["rotation"] = [0, vr[1], 0]

        return player, playmate, subject

    def split_items_into_receptacles_and_objects(self):
        """
        Split user items into receptacles (floor assets) and small objects.
        - For receptacles: build { type: {"count": n, "objects": [] } }
        - For small objects: map type/prefab to concrete prefab counts
        """
        items = self.scene_config.items
        odb = self.odb

        if not items:
            return {}, {}

        receptacle_object_counts: Dict[str, Any] = {}
        small_object_counts: Dict[str, int] = {}

        # Receptacle types are keys in odb.RECEPTACLES; small object types are keys in odb.OBJECT_DICT
        for name, cnt in items.items():
            key = name.strip()
            type_key = key.lower()

            receptacles = set(odb.RECEPTACLES.keys())
            small_objects = set(odb.OBJECT_DICT.keys()) - receptacles

            if type_key in receptacles:
                receptacle_object_counts[type_key] = {"count": int(cnt), "objects": []}
                continue

            # Treat as small object type or prefab
            if type_key in small_objects:
                # Choose random prefab per instance
                for _ in range(int(cnt)):
                    prefab = random.choice(odb.OBJECT_DICT[type_key])
                    small_object_counts[prefab] = small_object_counts.get(prefab, 0) + 1
            elif key in odb.PREFABS:
                small_object_counts[key] = small_object_counts.get(key, 0) + int(cnt)
            else:
                # Unknown key; ignore silently to stay robust
                continue

        return receptacle_object_counts, small_object_counts

    def place_specified_objects(self, receptacle_object_counts):
        """place requested furniture randomly in single room"""
        room_id = self.scene_config.room_spec.spec.room_id
        room = self.rooms[room_id]
        odb = self.odb

        specified_object_types = set()
        specified_object_instances = []

        for receptacle, d in receptacle_object_counts.items():
            receptacle_type = receptacle
            receptacle = random.choice(odb.OBJECT_DICT[receptacle.lower()])
            specified_object_types.add(odb.OBJECT_TO_TYPE[receptacle])
            prefab_size = odb.PREFABS[receptacle]["size"]
            count = d["count"]
            
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                rectangle = room.sample_next_rectangle()
                minx, minz, maxx, maxz = rectangle
                rotation = self.prefab_fit_rectangle(prefab_size, rectangle)

                if rotation == -1:
                    continue
                else:
                    x_size = (prefab_size["x"] if rotation == 0 else prefab_size["z"])
                    z_size = (prefab_size["z"] if rotation == 0 else prefab_size["x"])

                    minx += x_size / 2 + WALL_THICKNESS
                    minz += z_size / 2 + WALL_THICKNESS
                    maxx -= x_size / 2 + WALL_THICKNESS
                    maxz -= z_size / 2 + WALL_THICKNESS
                    x = np.random.uniform(minx, maxx)
                    z = np.random.uniform(minz, maxz)
                    bbox = (x - x_size / 2, z - z_size / 2, x + x_size / 2, z + z_size / 2)
                    if self.placer.place_rectangle(receptacle, bbox):
                        specified_object_instances.append(
                            {
                                "prefab": receptacle,
                                "position": [x, prefab_size["y"] / 2, z],
                                "rotation": [0, rotation, 0],
                                "scale": [1, 1, 1],
                                "parent": -1,
                                "type": "receptacle",
                                "room_id": room_id,
                                "is_receptacle": True,
                                "receptacle_type": receptacle_type,
                            }
                        )
                        count -= 1
                    if count == 0:
                        break
        return specified_object_types, specified_object_instances

    def add_other_objects(self, max_floor_objects):
        room_id = self.scene_config.room_spec.spec.room_id
        room = self.rooms[room_id]
        odb = self.odb

        spawnable_asset_group_info = self.get_spawnable_asset_group_info()
        
        asset = None
        spawnable_asset_groups = spawnable_asset_group_info[spawnable_asset_group_info[f"in{room.room_type}s"] > 0]

        floor_types, spawnable_assets = odb.FLOOR_ASSET_DICT[(room.room_type, room.split)]

        priority_asset_types = copy.deepcopy(odb.PRIORITY_ASSET_TYPES.get(room.room_type, []))

        for i in range(max_floor_objects):
            cache_rectangles = i != 0 and asset is None

            if cache_rectangles:
                # NOTE: Don't resample failed rectangles
                # room.last_rectangles.remove(rectangle)
                rectangle = room.sample_next_rectangle(cache_rectangles=True)
            else:
                rectangle = room.sample_next_rectangle()

            if rectangle is None:
                break

            x_info, z_info, anchor_delta, anchor_type = room.sample_anchor_location(rectangle)

            asset = self.sample_and_add_floor_asset(
                room=room,
                rectangle=rectangle,
                anchor_type=anchor_type,
                anchor_delta=anchor_delta,
                spawnable_assets=spawnable_assets,
                spawnable_asset_groups=spawnable_asset_groups,
                priority_asset_types=priority_asset_types,
                odb=odb,
            )

            if asset is None:
                continue

            # log(f'asset: {asset}')
            room.sample_place_asset_in_rectangle(
                asset=asset,
                rectangle=rectangle,
                anchor_type=anchor_type,
                x_info=x_info,
                z_info=z_info,
                anchor_delta=anchor_delta,
            )

            added_asset_types = []
            if "assetType" in asset:
                added_asset_types.append(asset["assetType"])
            else:
                added_asset_types.extend([o["assetType"] for o in asset["objects"]])

                if not asset["allowDuplicates"]:
                    spawnable_asset_groups = spawnable_asset_groups.query(f"assetGroupName!='{asset['assetGroupName']}'")

            for asset_type in added_asset_types:
                # Remove spawned object types from `priority_asset_types` when appropriate
                if asset_type in priority_asset_types:
                    priority_asset_types.remove(asset_type)

                allow_duplicates_of_asset_type = odb.PLACEMENT_ANNOTATIONS.loc[asset_type.lower()]["multiplePerRoom"]

                if not allow_duplicates_of_asset_type:
                    # NOTE: Remove all asset groups that have the type
                    spawnable_asset_groups = spawnable_asset_groups[
                        ~spawnable_asset_groups[f"has{asset_type.lower()}"]
                    ]

                    # NOTE: Remove all standalone assets that have the type
                    spawnable_assets = spawnable_assets[
                        spawnable_assets["assetType"] != asset_type
                    ]
        
        return spawnable_asset_groups, spawnable_assets
    
    def place_other_objects(self, specified_object_types, current_num_assets = 0):
        room_id = self.scene_config.room_spec.spec.room_id
        room = self.rooms[room_id]
        odb = self.odb

        object_instances = []
        
        for a in room.assets[current_num_assets:]:
            if isinstance(a, Asset):
                prefab = a.asset_id
                prefab_size = odb.PREFABS[prefab]["size"]
                bbox = (
                    (
                        a.position["x"] - prefab_size["x"] / 2,
                        a.position["z"] - prefab_size["z"] / 2,
                        a.position["x"] + prefab_size["x"] / 2,
                        a.position["z"] + prefab_size["z"] / 2,
                    )
                    if a.rotation == 0 or a.rotation == 180
                    else (
                        a.position["x"] - prefab_size["z"] / 2,
                        a.position["z"] - prefab_size["x"] / 2,
                        a.position["x"] + prefab_size["z"] / 2,
                        a.position["z"] + prefab_size["x"] / 2,
                    )
                )

                if not self.placer.place_rectangle(prefab, bbox):
                    log(f"Failed to place{prefab} into {bbox}")
                elif odb.OBJECT_TO_TYPE[prefab] in specified_object_types:
                    log("conflicted with specified objects!")
                else:
                    log(f"Placed {prefab} into position:{ format(a.position['x'],'.4f')},{format(a.position['z'],'.4f')}, bbox:{bbox} rotation:{a.rotation}")

                    is_receptacle = True
                    object_instances.append(
                        {
                            "prefab": a.asset_id,
                            "position": convert_vector(a.position),
                            "rotation": [0, a.rotation, 0],
                            "scale": [1, 1, 1],
                            "parent": room_id,
                            "type": ("interactable"if a.asset_id in self.odb.KINETIC_AND_INTERACTABLE_INFO["interactable_names"] else "kinematic"),
                            "room_id": room_id,
                            "is_receptacle": is_receptacle,
                        }
                    )

            else:  # is asset_group
                assets_dict = a.assets_dict
                max_bbox = (10000, 100000, -1, -1)
                asset_group_full_name = []

                conflict = False
                for asset in assets_dict:
                    prefab = asset["assetId"]
                    asset_type = odb.OBJECT_TO_TYPE[prefab]
                    if asset_type in specified_object_types:
                        conflict = True
                        break

                    asset_group_full_name.append(prefab)
                    if "children" in asset:
                        for child in asset["children"]:
                            prefab = child["assetId"]
                            asset_group_full_name.append(prefab)
                    prefab_size = odb.PREFABS[prefab]["size"]
                    bbox = (
                        (
                            asset["position"]["x"] - prefab_size["x"] / 2,
                            asset["position"]["z"] - prefab_size["z"] / 2,
                            asset["position"]["x"] + prefab_size["x"] / 2,
                            asset["position"]["z"] + prefab_size["z"] / 2,
                        )
                        if asset["rotation"] == 0 or asset["rotation"] == 180
                        else (
                            asset["position"]["x"] - prefab_size["z"] / 2,
                            asset["position"]["z"] - prefab_size["x"] / 2,
                            asset["position"]["x"] + prefab_size["z"] / 2,
                            asset["position"]["z"] + prefab_size["x"] / 2,
                        )
                    )
                    max_bbox = (
                        min(max_bbox[0], bbox[0]),
                        min(max_bbox[1], bbox[1]),
                        max(max_bbox[2], bbox[2]),
                        max(max_bbox[3], bbox[3]),
                    )
                asset_group_full_name = "+".join(asset_group_full_name)

                if not self.placer.place_rectangle(asset_group_full_name, max_bbox):
                    log(f"Failed to place{asset_group_full_name} into {max_bbox}")
                elif conflict:
                    log("conflicted with specified objects!")
                else:

                    log(f"Placed {asset_group_full_name} into {max_bbox}")
                    for asset in assets_dict:

                        is_receptacle = True
                        if (
                            "tv" in asset["assetId"].lower()
                            or "chair" in asset["assetId"].lower()
                        ):
                            is_receptacle = False

                        object_instances.append(
                            {
                                "prefab": asset["assetId"],
                                "position": (
                                    asset["position"]["x"],
                                    asset["position"]["y"],
                                    asset["position"]["z"],
                                ),
                                "rotation": [0, asset["rotation"]["y"], 0],
                                "scale": [1, 1, 1],
                                "parent": 0,  # 0 represents the floor
                                "type": (
                                    "interactable"
                                    if asset["assetId"]
                                    in self.odb.KINETIC_AND_INTERACTABLE_INFO[
                                        "interactable_names"
                                    ]
                                    else "kinematic"
                                ),
                                "room_id": room_id,
                                "is_receptacle": is_receptacle,
                            }
                        )
                        if "children" in asset:
                            for child in asset["children"]:

                                is_receptacle = True
                                if (
                                    "tv" in asset["assetId"].lower()
                                    or "chair" in asset["assetId"].lower()
                                ):
                                    is_receptacle = False

                                object_instances.append(
                                    {
                                        "prefab": child["assetId"],
                                        "position": (
                                            child["position"]["x"],
                                            child["position"]["y"],
                                            child["position"]["z"],
                                        ),
                                        "rotation": [0, child["rotation"]["y"], 0],
                                        "scale": [1, 1, 1],
                                        "parent": 0,  # 0 represents the floor
                                        "type": (
                                            "interactable"
                                            if child["assetId"]
                                            in self.odb.KINETIC_AND_INTERACTABLE_INFO[
                                                "interactable_names"
                                            ]
                                            else "kinematic"
                                        ),
                                        "room_id": room_id,
                                        "is_receptacle": is_receptacle,
                                    }
                                )

        return object_instances

    def generate(self):
        odb = self.odb
        prefabs = odb.PREFABS
        room_spec = self.scene_config.room_spec

        # 1. Generate house structure
        house_structure = self.generate_structure()
        interior = house_structure.interior
        x_size = interior.shape[0]
        z_size = interior.shape[1]

        min_x, min_z, max_x, max_z = (0, 0, x_size * self.unit_size, z_size * self.unit_size)
        room_area = max_x * max_z

        self.placer = RectPlacer((min_x, min_z, max_x, max_z))

        max_floor_objects = max(7, int(room_area / 5))

        # 2. Add floors and walls
        floor_instances, floors = self.add_floors_and_walls(house_structure, room_spec, odb, prefabs)
        # add light
        # light_prefab = "LowPolyInterior2_Light_04"
        # light_y_size = prefabs[light_prefab]["size"]["y"]
        # floor_instances.append(
        #     {
        #         "prefab": "LowPolyInterior2_Light_04",
        #         "position": [max_x / 2, 3 - light_y_size / 2, max_z / 2],
        #         "rotation": [0, 0, 0],
        #         "scale": [1, 1, 1],
        #         "type": "kinematic",
        #     }
        # )
        
        # 3. Room Initialization
        floor_polygons = self.get_floor_polygons(house_structure.xz_poly_map)
        self.get_rooms(room_spec=room_spec, floor_polygons=floor_polygons)

        # 4. Add human and agent
        player, agent, subject = self.add_player_agent_subject(floors)

        # player = {
        #     "prefab": "",
        #     "position": [10, 0.05, 10],
        #     "rotation": [0, np.random.uniform(0, 360), 0],
        #     "scale": [1, 1, 1],
        #     "parent": -1,
        #     "type": "",
        # }
        # if room_num == 1:
        #     flag, success_agent = self.add_corner_agent(max_x, max_z)
        #     if flag:
        #         agent = success_agent
        

        # user specified object placement logic
        # 5. Place specified objects
        receptacle_object_counts, small_object_counts = self.split_items_into_receptacles_and_objects()
        specified_object_types = set()
        specified_object_instances = []
        if receptacle_object_counts:
            specified_object_types, specified_object_instances = self.place_specified_objects(receptacle_object_counts)
        
        # 6. prepare room assets
        object_instances = []
        spawnable_asset_groups, spawnable_assets = self.add_other_objects(max_floor_objects=max_floor_objects)

        # 7. place other objects
        object_instances = self.place_other_objects(specified_object_types)

        # 8. add small objects
        max_object_types_per_room = max_floor_objects * 3
        small_object_instances = []
        small_object_instances = add_small_objects(
            object_instances,
            odb,
            self.rooms,
            max_object_types_per_room,
            (min_x, min_z, max_x, max_z),
            object_counts=small_object_counts,
            specified_object_instances=specified_object_instances,
            receptacle_object_counts=receptacle_object_counts
        )

        # 9. Prepare scene
        instances = (floor_instances + subject + specified_object_instances + object_instances + small_object_instances)
                

        DEBUG = False
        if DEBUG:
            for inst in instances:
                inst["type"] = "kinematic"

        height = max(12, (max_z - min_z) * 1 + 2)
        log(f"min_x: {min_x}, max_x: {max_x}, min_z: {min_z}, max_z: {max_z}")
        center = [(min_x + max_x) / 2, height, (min_z + max_z) / 2]

        room_polygon = []
        for room in self.rooms.values():
            id = room.room_id
            polygon = list(room.room_polygon.polygon.exterior.coords)
            x_center = sum([x for x, _ in polygon]) / len(polygon)
            z_center = sum([z for _, z in polygon]) / len(polygon)
            x_size = max([x for x, _ in polygon]) - min([x for x, _ in polygon])
            z_size = max([z for _, z in polygon]) - min([z for _, z in polygon])
            room_polygon.append({
                'room_id':id,
                'room_type': room.room_type,
                'position': [x_center, 1.5, z_center],
                'size': [x_size,3, z_size],
                'polygon':polygon
            })
            # print(f'room {id} polygon: {polygon}')

        infos = {
            "prompt": "",
            "instances": instances,
            "player": player,
            "agent": agent,
            "center": center,
            "room_polygon": room_polygon,
        }

        # with open("last_scene.json", "w", encoding="utf-8") as f:
        #     json.dump(infos, f, ensure_ascii=False, indent=4)
        return SceneBundle(self, infos, max_floor_objects, spawnable_asset_groups, spawnable_assets, specified_object_types, max_object_types_per_room)
