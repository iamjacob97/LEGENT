import json
import random
import math
import numpy as np
from dataclasses import dataclass
from attrs import Attribute, define, field
from typing import Dict, List, Optional, Tuple, Any, Union, Literal, Callable, Set

from esha.esha_helpers import log
from esha.esha_house import ESHARoom, ESHARoomSpec, ESHAHouse, generate_house_structure
from legent.scene_generation.room import Room
from legent.scene_generation.generator import HouseGenerator
from legent.server.rect_placer import RectPlacer
from legent.scene_generation.objects import ObjectDB, get_default_object_db
from legent.scene_generation.constants import OUTDOOR_ROOM_ID


RoomType = str  # "Bedroom" | "LivingRoom" | "Kitchen" | "Bathroom"

@dataclass
class SceneConfig:
    dims: Tuple[int, int] = None # (x_size, z_size) in cell units
    include_other_items: bool = True  # include LEGENT proc objects in addition to specified items
    items: Optional[Dict[str, int]] = None  # user-specified dictionary; keys are LEGENT types (e.g., "orange", "table")

DEFAULT_FLOOR_SIZE = 2.5
DEFAULT_WALL_PREFAB = "LowPolyInterior2_Wall1_C1_01"
class ESHAGenerator(HouseGenerator):
    def __init__(
        self, 
        room_spec: ESHARoomSpec = ESHARoomSpec(room_spec_id="ESHA-SingleRoom", spec=ESHARoom(room_id=3, room_type="LivingRoom")), 
        scene_config: SceneConfig = SceneConfig(),
        objectDB: ObjectDB = get_default_object_db(), 
        unit_size=2.5,
    ) -> None:
        self.room_spec = room_spec
        self.scene_config = scene_config
        self.objectDB = objectDB
        self.rooms: Dict[str, Room] = dict()
        self.unit_size = unit_size
        self.half_unit_size = unit_size / 2  # Half of the size of a unit in the grid
        self.scale_ratio = unit_size / DEFAULT_FLOOR_SIZE

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"room_spec={self.room_spec!r}, "
                f"scene_config={self.scene_config!r}, "
                f"objectDB={type(self.objectDB).__name__}, "
                f"rooms={len(self.rooms)}, "
                f"unit_size={self.unit_size}, "
                f"scale_ratio={self.scale_ratio:.2f})") 
    
    

    def generate_structure(self, room_spec: ESHARoomSpec) -> ESHAHouse:
        house_structure = generate_house_structure(room_spec=room_spec, scene_config=self.scene_config)
        return house_structure
    
    def add_floors_and_walls(self, house_structure, room_spec, odb, prefabs, 
                            add_ceiling = False, remove_out_walls = False):
        """ESHA simplified version - single room with no doors."""
        room_id = room_spec.spec.room_id
        room2wall = {0: DEFAULT_WALL_PREFAB, room_id: np.random.choice(odb.MY_OBJECTS["wall"][:])}
        wall_y_size = prefabs[room2wall[room_id]]["size"]["y"]
        floor_prefab = np.random.choice(odb.MY_OBJECTS["floor"])
        floor_x_size, floor_y_size, floor_z_size = (
            prefabs[floor_prefab]["size"]["x"],
            prefabs[floor_prefab]["size"]["y"],
            prefabs[floor_prefab]["size"]["z"],
        )
        
        log(f"floor_size: {floor_x_size}x{floor_y_size}x{floor_z_size}")
        
        floors = house_structure.floorplan
        floors = np.where(floors == 1, 0, floors)
        log(f"floors:\n{floors}")
        
        floor_instances = []
        
        # Generate floors and walls
        for i in range(floors.shape[0]):
            for j in range(floors.shape[1]):
                current_cell = floors[i][j]
                
                # Place floor tiles in room areas (not wall areas)
                if current_cell == room_id:  # Room interior
                    x = (i + 0.5 - 1) * self.unit_size
                    z = (j + 0.5 - 1) * self.unit_size
                    
                    # Floor tile
                    floor_instances.append({
                        "prefab": floor_prefab,
                        "position": [x, -floor_y_size / 2, z],
                        "rotation": [0, 90, 0],
                        "scale": [self.scale_ratio, 1, self.scale_ratio],
                        "type": "kinematic",
                    })
                    
                    # Ceiling (optional)
                    if add_ceiling:
                        floor_instances.append({
                            "prefab": floor_prefab,
                            "position": [x, wall_y_size + floor_y_size / 2, z],
                            "rotation": [0, 90, 0],
                            "scale": [self.scale_ratio, 1, self.scale_ratio],
                            "type": "kinematic",
                        })


                WALL_PREFAB = room2wall[current_cell]
                wall_x_size, wall_y_size, wall_z_size = (
                    prefabs[WALL_PREFAB]["size"]["x"],
                    prefabs[WALL_PREFAB]["size"]["y"],
                    prefabs[WALL_PREFAB]["size"]["z"],
                )
                
                # Place walls at boundaries
                # Vertical walls (X direction)
                if i < floors.shape[0] - 1:
                    next_cell = floors[i + 1][j]
                    if remove_out_walls and (current_cell==0 or next_cell==0):
                        continue

                    if current_cell != next_cell:  # Boundary detected
                        x = (i + 1 - 1) * self.unit_size
                        z = (j + 0.5 - 1) * self.unit_size
                        
                        left_x = x - wall_z_size / 4
                        right_x = x + wall_z_size / 4

                        left_wall_prefab = room2wall[current_cell]
                        right_wall_prefab = room2wall[next_cell]

                        scale = [self.scale_ratio, 1, 0.5]
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

                    if current_cell != 0:
                        floor_instances.append(left_wall)
                    if next_cell != 0:
                        floor_instances.append(right_wall)

                
                # Horizontal walls (Z direction)
                if j < floors.shape[1] - 1:
                    next_cell = floors[i][j + 1]
                    if remove_out_walls and (current_cell==0 or next_cell==0):
                        continue

                    if current_cell != next_cell:  # Boundary detected
                        x = (i + 0.5 - 1) * self.unit_size
                        z = (j + 1 - 1) * self.unit_size

                        up_z = z - wall_z_size / 4
                        down_z = z + wall_z_size / 4

                        up_wall_prefab = room2wall[current_cell]
                        down_wall_prefab = room2wall[next_cell]

                        scale = [self.scale_ratio, 1, 0.5]
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
                    if current_cell != 0:
                        floor_instances.append(up_wall)
                    if next_cell != 0:
                        floor_instances.append(down_wall)
        
        return floor_instances, floors 
    
    def get_rooms(self, room_spec: ESHARoomSpec, floor_polygons):
        room_id = room_spec.spec.room_id
        room_type = room_spec.spec.room_type
        polygon = floor_polygons[f"room|{room_id}"]
        room = Room(
            polygon=polygon,
            room_type=room_type,
            room_id=room_id,
            odb=self.odb,
        )
        self.rooms[room_id] = room

    def generate(
        self,
        object_counts: Dict[str, int] = {},
        receptacle_object_counts: Dict[str, Dict[str, int]] = {}
    ):
        odb = self.odb
        prefabs = odb.PREFABS
        room_spec = self.room_spec


        house_structure = self.generate_structure(room_spec=room_spec)
        interior = house_structure.interior
        x_size = interior.shape[0]
        z_size = interior.shape[1]

        min_x, min_z, max_x, max_z = (
            0,
            0,
            x_size * self.unit_size,
            z_size * self.unit_size,
        )
        self.placer = RectPlacer((min_x, min_z, max_x, max_z))

        floor_instances, floors = self.add_floors_and_walls(
            house_structure, room_spec, odb, prefabs, add_ceiling=True, remove_out_walls=False
        )
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

        floor_polygons = self.get_floor_polygons(house_structure.xz_poly_map)

        self.get_rooms(room_spec=room_spec, floor_polygons=floor_polygons)

        player, agent = self.add_human_and_agent(floors)
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

        max_floor_objects = 10

        spawnable_asset_group_info = self.get_spawnable_asset_group_info()

        specified_object_instances = []
        specified_object_types = set()
        if receptacle_object_counts:
            # first place the specified receptacles
            for receptacle, d in receptacle_object_counts.items():
                receptacle_type = receptacle
                receptacle = random.choice(odb.OBJECT_DICT[receptacle.lower()])
                specified_object_types.add(odb.OBJECT_TO_TYPE[receptacle])
                count = d["count"]
                prefab_size = odb.PREFABS[receptacle]["size"]
                for _ in range(MAX_SPECIFIED_NUMBER):
                    success_flag = False
                    for room in self.rooms.values():
                        for _ in range(MAX_SPECIFIED_RECTANGLE_RETRIES):
                            rectangle = room.sample_next_rectangle()
                            minx, minz, maxx, maxz = rectangle
                            rect_x = maxx - minx
                            rect_z = maxz - minz
                            rotation = self.prefab_fit_rectangle(prefab_size, rectangle)

                            if rotation == -1:
                                continue
                            else:
                                x_size = (
                                    prefab_size["x"] if rotation == 0 else prefab_size["z"]
                                )
                                z_size = (
                                    prefab_size["z"] if rotation == 0 else prefab_size["x"]
                                )
                                minx += x_size / 2 + WALL_THICKNESS
                                minz += z_size / 2 + WALL_THICKNESS
                                maxx -= x_size / 2 + WALL_THICKNESS
                                maxz -= z_size / 2 + WALL_THICKNESS
                                x = np.random.uniform(minx, maxx)
                                z = np.random.uniform(minz, maxz)
                                bbox = (
                                    x - x_size / 2,
                                    z - z_size / 2,
                                    x + x_size / 2,
                                    z + z_size / 2,
                                )
                                if self.placer.place_rectangle(receptacle, bbox):
                                    specified_object_instances.append(
                                        {
                                            "prefab": receptacle,
                                            "position": [x, prefab_size["y"] / 2, z],
                                            "rotation": [0, rotation, 0],
                                            "scale": [1, 1, 1],
                                            "parent": -1,
                                            "type": "receptacle",
                                            "room_id": room.room_id,
                                            "is_receptacle": True,
                                            "receptacle_type": receptacle_type,
                                        }
                                    )
                                    log(
                                        f"Specified {receptacle} into position:{ format(x,'.4f')},{format(z,'.4f')}, bbox:{bbox} rotation:{rotation}"
                                    )
                                    success_flag = True
                                    count -= 1
                                    break
                        if success_flag:
                            break
                    if count == 0:
                        break

        object_instances = []
        for room in self.rooms.values():
            asset = None
            spawnable_asset_groups = spawnable_asset_group_info[
                spawnable_asset_group_info[f"in{room.room_type}s"] > 0
            ]

            floor_types, spawnable_assets = odb.FLOOR_ASSET_DICT[
                (room.room_type, room.split)
            ]

            priority_asset_types = copy.deepcopy(
                odb.PRIORITY_ASSET_TYPES.get(room.room_type, [])
            )
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

                x_info, z_info, anchor_delta, anchor_type = room.sample_anchor_location(
                    rectangle
                )

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
                        spawnable_asset_groups = spawnable_asset_groups.query(
                            f"assetGroupName!='{asset['assetGroupName']}'"
                        )

                for asset_type in added_asset_types:
                    # Remove spawned object types from `priority_asset_types` when appropriate
                    if asset_type in priority_asset_types:
                        priority_asset_types.remove(asset_type)

                    allow_duplicates_of_asset_type = odb.PLACEMENT_ANNOTATIONS.loc[
                        asset_type.lower()
                    ]["multiplePerRoom"]

                    if not allow_duplicates_of_asset_type:
                        # NOTE: Remove all asset groups that have the type
                        spawnable_asset_groups = spawnable_asset_groups[
                            ~spawnable_asset_groups[f"has{asset_type.lower()}"]
                        ]

                        # NOTE: Remove all standalone assets that have the type
                        spawnable_assets = spawnable_assets[
                            spawnable_assets["assetType"] != asset_type
                        ]

        def convert_position(position: Vector3):
            x = a.position["x"]
            y = a.position["y"]
            z = a.position["z"]
            return (x, y, z)

        for room in self.rooms.values():

            log(f"room: {room.room_id}")
            for a in room.assets:
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
                        log(f"conflicted with specified objects!")
                    else:
                        log(
                            f"Placed {prefab} into position:{ format(a.position['x'],'.4f')},{format(a.position['z'],'.4f')}, bbox:{bbox} rotation:{a.rotation}"
                        )

                        is_receptacle = True
                        object_instances.append(
                            {
                                "prefab": a.asset_id,
                                "position": convert_position(a.position),
                                "rotation": [0, a.rotation, 0],
                                "scale": [1, 1, 1],
                                "parent": room.room_id,
                                "type": (
                                    "interactable"
                                    if a.asset_id
                                    in self.odb.KINETIC_AND_INTERACTABLE_INFO[
                                        "interactable_names"
                                    ]
                                    else "kinematic"
                                ),
                                "room_id": room.room_id,
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
                        log(f"conflicted with specified objects!")
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
                                    "room_id": room.room_id,
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
                                            "room_id": room.room_id,
                                            "is_receptacle": is_receptacle,
                                        }
                                    )

        max_object_types_per_room = 10
        small_object_instances = []
        small_object_instances = add_small_objects(
            object_instances,
            odb,
            self.rooms,
            max_object_types_per_room,
            (min_x, min_z, max_x, max_z),
            object_counts=object_counts,
            specified_object_instances=specified_object_instances,
            receptacle_object_counts=receptacle_object_counts,
        )

        ### STEP 5: Adjust Positions for Unity GameObject
        # Convert all the positions (the center of the mesh bounding box) to positions of Unity GameObject transform
        # They are not equal because position of a GameObject also depends on the relative center offset of the mesh within the prefab

        instances = (
            floor_instances
            + object_instances
            + specified_object_instances
            + small_object_instances
        )

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
        with open("last_scene.json", "w", encoding="utf-8") as f:
            json.dump(infos, f, ensure_ascii=False, indent=4)
        return infos



        
        



