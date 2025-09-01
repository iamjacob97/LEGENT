from typing import Dict, Any, Optional, Tuple
import random

from legent.scene_generation.objects import ObjectDB, get_default_object_db

def log(*args, **kwargs):
    pass

#------------Scene Parsing------------#
def split_items_into_receptacles_and_objects(
        odb: ObjectDB = get_default_object_db(), 
        items: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Split user items into receptacles (floor assets) and small objects.
    - For receptacles: build { type: {"count": n, "objects": [] } }
    - For small objects: map type/prefab to concrete prefab counts
    """
    if not items:
        return {}, {}

    receptacle_object_counts: Dict[str, Any] = {}
    object_counts: Dict[str, int] = {}

    # Receptacle types are keys in odb.RECEPTACLES; small object types are keys in odb.OBJECT_DICT
    for name, cnt in items.items():
        key = name.strip()
        type_key = key.lower()

        if type_key in odb.RECEPTACLES:
            receptacle_object_counts[type_key] = {"count": int(cnt), "objects": []}
            continue

        # Treat as small object type or prefab
        if type_key in odb.OBJECT_DICT:
            # Choose random prefab per instance
            for _ in range(int(cnt)):
                prefab = random.choice(odb.OBJECT_DICT[type_key])
                object_counts[prefab] = object_counts.get(prefab, 0) + 1
        elif key in odb.PREFABS:
            object_counts[key] = object_counts.get(key, 0) + int(cnt)
        else:
            # Unknown key; ignore silently to stay robust
            continue

    return receptacle_object_counts, object_counts