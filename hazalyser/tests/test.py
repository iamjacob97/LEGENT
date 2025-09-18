import random
from hazalyser import SceneConfig, Controller, HazardRoom, HazardRoomSpec

room_spec = HazardRoomSpec(room_spec_id="ESHA-SingleRoom", 
                           spec=HazardRoom(room_id=1, room_type=random.choice(["Kitchen", "LivingRoom", "Bedroom", "Bathroom"])))
scene_config = SceneConfig(room_spec = room_spec, dims = None, subject="oldman_ernest.glb")
controller = Controller(scene_config)

controller.start()
