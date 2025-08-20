from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Literal, Callable
from attrs import define, field
import numpy as np

from legent.scene_generation.constants import OUTDOOR_ROOM_ID


class ESHARoom:
    def __init__(
        self,
        room_id: int,
        room_type: Optional[Literal["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]],
    ):
        assert room_type in {"Kitchen", "LivingRoom", "Bedroom", "Bathroom", None}
        if room_id in {0, OUTDOOR_ROOM_ID}:
            raise Exception(f"room_id of 0 and {OUTDOOR_ROOM_ID} are reserved!")

        self.room_id = room_id
        self.room_type = room_type

    def __repr__(self):
        return f"ESHARoom(room_id={self.room_id}, room_type={self.room_type})"

    def __str__(self):
        return self.__repr__()

@define
class ESHARoomSpec:
    room_spec_id: str
    spec: ESHARoom

    dims: Optional[Callable[[], Tuple[int, int]]] = None
    """The (x_size, z_size) dimensions of the house.

    Note that this size will later be scaled up by interior_boundary_scale.
    """
     
@define
class ESHAHouse:
    interior: np.ndarray
    floorplan: np.ndarray
    connectors: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    boundary: Dict[Tuple[int, int], Set[Tuple[Tuple[float, float], Tuple[float, float]]]]
    xz_poly_map: Dict[int, List[Tuple[float, float]]]
    ceiling_height: float

#----------house_builders----------#
def generate_house_structure(room_spec, scene_config, unit_size = 2.5):
    dims = scene_config.dims

    generate_dims = None
    if dims is not None:
        generate_dims = dims
    elif room_spec.dims is not None:
        generate_dims = room_spec.dims()

    interior = generate_interior(room_spec=room_spec, dims=generate_dims)
    floorplan = np.pad(interior, pad_width=1, mode="constant", constant_values=OUTDOOR_ROOM_ID)
    connectors = find_connectors(floorplan)
    boundary = find_walls(interior, connectors) # TODO: add room_id to walls
    boundary = scale_boundary(boundary, scale=unit_size)
    xz_poly_map = get_xz_poly_map(boundary, room_spec.spec.room_id)

    ceiling_height = 0
    return ESHAHouse(
        interior=interior,
        floorplan=floorplan,
        connectors=connectors,
        boundary=boundary,
        xz_poly_map=xz_poly_map,
        ceiling_height=ceiling_height,
    )
    
def generate_interior(room_spec, dims: Optional[Tuple[int, int]] = None) -> np.array:
    if dims is None:
            x_size = np.random.randint(low=2, high=4)
            z_size = np.random.randint(low=2, high=4)
    else:
        x_size, z_size = dims

    interior = np.full((z_size, x_size), room_spec.spec.room_id, dtype=int)
    
    return interior
    
def find_connectors(floorplan: np.array):
    connectors = defaultdict(list)
    for row in range(len(floorplan) - 1):
        for col in range(len(floorplan[0]) - 1):
            a = floorplan[row, col]
            b = floorplan[row, col + 1]
            if a != b:
                connectors[(int(min(a, b)), int(max(a, b)))].append(
                    ((row - 1, col), (row, col))
                )
            b = floorplan[row + 1, col]
            if a != b:
                connectors[(int(min(a, b)), int(max(a, b)))].append(
                    ((row, col - 1), (row, col))
                )
    return connectors

def find_walls(interior, connectors):
    """Find walls for ESHA single room - optimized version."""
    boundary = {}
    if len(connectors) != 1:
        raise ValueError("Expected 1 wall group")
    wall_group_id = next(iter(connectors))

    rows, cols = interior.shape
    walls = set()
    # Horizontal walls
    walls.add(((0, 0), (0, cols)))      # top
    walls.add(((rows, 0), (rows, cols)))  # bottom

    # Vertical walls
    walls.add(((0, 0), (rows, 0)))      # left
    walls.add(((0, cols), (rows, cols)))  # right

    boundary = {wall_group_id: walls}

    return boundary

def scale_boundary(boundary, scale, precision: int = 3):
    """Scale the boundary of the house by a factor of scale."""
    out = dict()
    for key, lines in boundary.items():
        scaled_lines = set()
        for (x0, z0), (x1, z1) in lines:
            scaled_lines.add(
                (
                    (round(x0 * scale, precision), round(z0 * scale, precision)),
                    (round(x1 * scale, precision), round(z1 * scale, precision)),
                )
            )
        out[key] = scaled_lines
    return out

def get_xz_poly_map(boundary, room_id):
    """Get the xz_poly_map for the house."""
    WALL_THICKNESS = 0.1

    room_wall_loop = get_wall_loop(list(boundary.values())[0])

        # determines if the loop is counter-clockwise, flips if it is
    edge_sum = 0
    for (x0, z0), (x1, z1) in room_wall_loop:
        dist = x0 * z1 - x1 * z0
        edge_sum += dist
    if edge_sum > 0:
        room_wall_loop = [(p1, p0) for p0, p1 in reversed(room_wall_loop)]

    points = []
    dirs = []
    for p0, p1 in room_wall_loop:
        points.append(p0)
        if p1[0] > p0[0]:
            dirs.append("right")
        elif p1[0] < p0[0]:
            dirs.append("left")
        elif p1[1] > p0[1]:
            dirs.append("up")
        elif p1[1] < p0[1]:
            dirs.append("down")

    for i in range(len(points)):
        this_dir = dirs[i]
        last_dir = dirs[i - 1] if i > 0 else dirs[-1]
        if this_dir == "right" and last_dir == "up":
            points[i] = (
                points[i][0] + WALL_THICKNESS,
                points[i][1] - WALL_THICKNESS,
            )
        elif this_dir == "down" and last_dir == "right":
            points[i] = (
                points[i][0] - WALL_THICKNESS,
                points[i][1] - WALL_THICKNESS,
            )
        elif this_dir == "left" and last_dir == "down":
            points[i] = (
                points[i][0] - WALL_THICKNESS,
                points[i][1] + WALL_THICKNESS,
            )
        elif this_dir == "up" and last_dir == "left":
            points[i] = (
                points[i][0] + WALL_THICKNESS,
                points[i][1] + WALL_THICKNESS,
            )
    room_wall_loop = list(zip(points, points[1:] + [points[0]]))
    return {room_id: room_wall_loop}


def get_wall_loop(walls):
    walls_left = walls.copy()
    start_wall = list(walls)[0]
    out = [start_wall]
    walls_left.remove(start_wall)
    while walls_left:
        for wall in walls_left:
            if out[-1][1] == wall[0]:
                out.append(wall)
                walls_left.remove(wall)
                break
            elif out[-1][1] == wall[1]:
                out.append((wall[1], wall[0]))
                walls_left.remove(wall)
                break
        else:
            raise Exception(f"No connecting wall for {out[-1]}!")
    return out

