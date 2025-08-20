from esha.esha_house import ESHARoom, ESHARoomSpec, generate_house_structure
from esha.esha_generator import SceneConfig

scene_config = SceneConfig()

room_spec = ESHARoomSpec(room_spec_id="SingleRoom", spec=ESHARoom(room_id=3, room_type="LivingRoom"), dims=lambda: (2, 4))
house = generate_house_structure(room_spec=room_spec, scene_config=scene_config, unit_size=2.5)
print(house)