import numpy as np

class Node:
    def _init_(self, id_num, type, nearest_nei_id_nums):
        self.id_num = id_num
        self.type = type
        assert type in ["X", "Y"]
        self.nearest_nei_id_nums = nearest_nei_id_nums
        self.probs = [.5, .5]
        self.mutual_info = 0

    def sample_node(self):
        return np.random.choice([0, 1], p=self.probs)