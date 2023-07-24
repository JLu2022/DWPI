import random
import numpy as np


class PreferenceSpace(object):
    def __init__(self, num_objective=2, granularity=100):
        self.num_objective = num_objective
        self.granularity = granularity
        # self.default_pref = default_pref

    def sample(self, default_pref=None):
        pref = []
        upper_bound = self.granularity + 1
        for _ in range(self.num_objective - 1):
            p = random.choice([x for x in range(0, upper_bound)])
            pref.append(p / self.granularity)
            upper_bound = self.granularity - p
        last_p = 1 - sum(pref)
        pref.append(last_p)

        preference = np.array(default_pref) if default_pref is not None else np.array(pref)
        return preference

    def sample_batch(self, batch_size=1, default_pref=None, threshold=1.0):
        pref_batch = []
        for i in range(batch_size):
            pref = self.sample(default_pref=default_pref)
            while pref[1] > threshold:
                pref = self.sample(default_pref=default_pref)
            pref_batch.append(pref)
        return np.array(pref_batch)


if __name__ == '__main__':
    preference_space = PreferenceSpace(num_objective=2, granularity=100)
    for i in range(10):
        # preference = preference_space.sample()
        # print(f"preference:{preference}")
        print(preference_space.sample_batch(batch_size=10,threshold=0.1))
