import numpy as np
from tensorflow import keras
from collections import deque  # Used for replay buffer and reward tracking


class ReplayMemory(deque):

    def sample(self, batch_size, max_batch_size=0, current_weight=None, current_state=None, NER_ratio=0.5):
        # print(f"len:{len(self)}")
        indices = np.random.randint(len(self), size=batch_size)
        batch = [self[index] for index in indices]
        states, actions, time_rewards, treasure_rewards, reward_scalar, next_states, dones, weightss = [
            np.array([experience[field_index] for experience in batch]) for field_index in range(8)]
        return states, actions, time_rewards, treasure_rewards, reward_scalar, next_states, dones, weightss


class NER_Memory(deque):
    """
        Inherits from the 'deque' class to add a method called 'sample' for
        sampling batches from the deque.
    """

    def sample(self, batch_size, max_batch_size=None, current_weight=None, current_state=None, NER_ratio=0.5):
        # Random sample of indices
        indices = np.random.randint(len(self), size=max_batch_size)
        # Filter the batch from the deque
        batch = []
        rank_dict = {}
        for index in indices:
            exp = self[index]
            # print("ext;",exp[-1])
            w = exp[-1]
            s = exp[0]
            similarity = self.weight_similarity(current_weight, w)
            rank_dict[index] = similarity
        sorted_similarity = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)
        while len(batch) < int(NER_ratio * batch_size):
            batch.append(self[sorted_similarity[len(batch)][0]])
        while len(batch) < batch_size:
            batch.append(self[np.random.choice(indices)])

        # Unpach and create numpy arrays for each element type in the batch
        s, a, r, next_s, dones, w = [np.array([exp[field_index] for exp in batch]) for field_index in range(6)]
        return s, a, r, next_s, dones, w

    def weight_similarity(self, current_weight, sampled_weight):
        mseTensor = keras.losses.mse(current_weight, sampled_weight)
        mse = mseTensor.numpy()
        return mse


if __name__ == '__main__':
    pass
