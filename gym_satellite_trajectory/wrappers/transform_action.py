import gym


class TransformAction(gym.ActionWrapper):
    def __init__(self, env, action_space, f, **kwargs):
        if isinstance(env, str):
            env = gym.make(env, **kwargs)
        super(gym.ActionWrapper, self).__init__(env)
        assert callable(f)
        self.action_space = action_space
        self.f = f

    def action(self, action):
        return self.f(action)

    def reverse_action(self, action):
        raise NotImplementedError
