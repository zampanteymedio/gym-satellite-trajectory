import gym


class TransformAction(gym.ActionWrapper):
    def __init__(self, env, action_space, f, f2=None, **kwargs):
        if isinstance(env, str):
            env = gym.make(env, **kwargs)
        super(gym.ActionWrapper, self).__init__(env)
        assert callable(f)
        if f2 is not None:
            assert callable(f2)
        self.action_space = action_space
        self.f = f
        self.f2 = f2

    def action(self, action):
        return self.f(action)

    def reverse_action(self, action):
        if self.f2 is None:
            raise NotImplementedError
        return self.f2(action)
