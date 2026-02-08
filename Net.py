import itertools
import numpy as np

from Cond_Prob import *
from Node import *
from globals import *


class Net:
    def __init__(self, beta, g, h, lam=1E10):
        self.cpt = Cond_Prob(beta, g, h, lam)
        self.x_nodes = []
        self.y_nodes = []
        self.arrows = []
        self.create_nodes()
        self.calc_y_node_probs()
        self.calc_y_node_mutual_infos()

    def get_nd_from_id(self, id_num, type):
        assert id_num-1 in range(NUM_DNODES)
        if type == "X":
            return self.x_nodes[id_num]
        elif type == "Y":
            return self.y_nodes[id_num]
        assert None, "this node type does not exist"



    def create_nodes(self):
        nd_id = 0
        for row in range(1, DGRAPH_NUM_ROWS+1):
            for col in range(1, DGRAPH_NUM_COLS+1):
                nd_id += 1
                row_res = row % DGRAPH_NUM_ROWS
                col_res = col % DGRAPH_NUM_ROWS
                if row_res==1 and col_res==1:
                    nearest_nei = [nd_id+1, nd_id + DGRAPH_NUM_COLS]
                elif row_res==1 and col_res==0:
                    nearest_nei = [nd_id-1, nd_id + DGRAPH_NUM_COLS]
                elif row_res == 0 and col_res == 1:
                    nearest_nei = [nd_id+1, nd_id - DGRAPH_NUM_COLS]
                elif row_res == 0 and col_res == 0:
                    nearest_nei = [nd_id -1 1, nd_id - DGRAPH_NUM_COLS]
                else:
                    nearest_nei = [nd_id+1, nd_id-1, nd_id + DGRAPH_NUM_COLS,
                          nd_id - DGRAPH_NUM_COLS]
                x_node = Node(nd_id, "X", nearest_nei)
                y_node = Node(nd_id, "Y", nearest_nei)
                self.x_nodes.append(x_node)
                self.y_nodes.append(y_node)

    def calc_y_node_probs(self):
        nd_id = 0
        for row in range(1, DGRAPH_NUM_ROWS):
            for col in range(1, DGRAPH_NUM_COLS):
                nd_id += 1
                y_nd = self.get_nd_from_id(nd_id, "Y")
                prob_m = 0
                prob_p = 0
                num_nearest_nei = len(y_nd.nearest_nei_id_nums)
                prob_nearest_nei = (0.5)**num_nearest_nei
                for nearest_nei_state  in itertools.product(
                        [-1,1], repeat=num_nearest_nei+1):
                    for x_spin in [-1, 1]:
                        cond_prob_m, cond_prob_p, zz =\
                            self.cpt.calc_cond_probs_y_if_abcd_x(
                                nearest_nei_state, x_spin)
                        prob_m += cond_prob_m*prob_nearest_nei*0.5
                        prob_p += cond_prob_p*prob_nearest_nei*0.5
                y_nd.probs= [prob_m, prob_p]

    def calc_y_node_mutual_infos(self):
        nd_id = 0
        for row in range(1, DGRAPH_NUM_ROWS):
            for col in range(1, DGRAPH_NUM_COLS):
                nd_id += 1
                y_nd = self.get_nd_from_id(nd_id, "Y")
                prob_m = 0
                prob_p = 0
                mutual_info = 0
                num_nearest_nei = len(y_nd.nearest_nei_id_nums)
                prob_nearest_nei = (0.5)**num_nearest_nei
                for nearest_nei_state  in itertools.product(
                        [-1,1], repeat=num_nearest_nei+1):
                    for x_spin in [-1, 1]:
                        cond_prob_m, cond_prob_p, zz =\
                            self.cpt.calc_cond_probs_y_if_abcd_x(
                                nearest_nei_state, x_spin)
                        prob_m += cond_prob_m*prob_nearest_nei*0.5
                        prob_p += cond_prob_p*prob_nearest_nei*0.5
                        mutual_info += prob_m * np.log(
                            cond_prob_m/y_nd.probs[0])
                        mutual_info += prob_p * np.log(
                            cond_prob_p / y_nd.probs[1])
                y_nd.mutual_info = mutual_info



