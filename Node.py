class Node:
    def _init_(self, id_num, type):
        self.id_num = id_num
        self.type = type
        assert type in ["nonroot", "top_root", "side_root"]
        self.probs_x = [0.0, 1.0]