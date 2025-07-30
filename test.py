from legent import Environment, Action, Observation

env = Environment(env_path="auto")
try:
    obs: Observation = env.reset()
    while True:
        action = Action()
        if obs.text != "":
            action.text = "I don't understand."
        obs = env.step(action)
finally:
    env.close()