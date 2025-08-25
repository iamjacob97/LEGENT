from esha.esha_generator import ESHAGenerator, SceneConfig
from legent import Environment, ResetInfo

generator = ESHAGenerator()

env = Environment(env_path="auto")

try:
    scene = generator.generate()
    env.reset(ResetInfo(scene=scene))
    while True:
        env.step()
finally:
    env.close()


