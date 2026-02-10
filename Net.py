import itertools
import numpy as np
from plotting import *

from Cond_Prob import *
from Node import *
from globals import *
from utils import *


class Net:
    def __init__(self, beta, jj, h):
        self.cpt = Cond_Prob(beta, jj, h)
        self.x_nodes = []
        self.y_nodes = []
        self.create_nodes()
        self.calc_y_node_probs()
        self.calc_y_node_mutual_infos()

    def get_nd_from_id(self, id_num, type):
        assert id_num - 1 in range(NUM_DNODES)
        if type == "X":
            return self.x_nodes[id_num - 1]
        elif type == "Y":
            return self.y_nodes[id_num - 1]
        assert None, "this node type does not exist"



    def create_nodes(self):
        nd_id = 0
        for row in range(1, DGRAPH_NUM_ROWS + 1):
            for col in range(1, DGRAPH_NUM_COLS + 1):
                nd_id += 1
                x_node = Node(nd_id, "X")
                y_node = Node(nd_id, "Y")
                self.x_nodes.append(x_node)
                self.y_nodes.append(y_node)

    def calc_y_node_probs(self):
        nd_id = 0
        for row in range(1, DGRAPH_NUM_ROWS):
            for col in range(1, DGRAPH_NUM_COLS):
                nd_id += 1
                y_nd = self.y_nodes[nd_id-1]
                prob_m = 0
                prob_p = 0
                num_nearest_nei = len(y_nd.nearest_nei)
                prob_nearest_nei = (0.5) ** num_nearest_nei
                for nearest_nei_states in itertools.product(
                        [-1, 1], repeat=num_nearest_nei):
                    for x_spin in [-1, 1]:
                        cond_prob_m, cond_prob_p, zz = \
                            self.cpt.calc_cond_probs_y_if_abcd_x(
                                nearest_nei_states, x_spin)
                        prob_m += cond_prob_m * prob_nearest_nei * 0.5
                        prob_p += cond_prob_p * prob_nearest_nei * 0.5
                y_nd.probs = [prob_m, prob_p]
                y_nd.entropy = coin_toss_entropy(prob_m)

    def calc_y_node_mutual_infos(self):
        nd_id = 0
        for row in range(1, DGRAPH_NUM_ROWS):
            for col in range(1, DGRAPH_NUM_COLS):
                nd_id += 1
                y_nd = self.get_nd_from_id(nd_id, "Y")
                mutual_info = 0
                num_nearest_nei = len(y_nd.nearest_nei)
                prob_nearest_nei = (0.5) ** num_nearest_nei
                for nearest_nei_state in itertools.product(
                        [-1, 1], repeat=num_nearest_nei):
                    for x_spin in [-1, 1]:
                        cond_prob_m, cond_prob_p, zz = \
                            self.cpt.calc_cond_probs_y_if_abcd_x(
                                nearest_nei_state, x_spin)
                        joint_prob_m = cond_prob_m * prob_nearest_nei * 0.5
                        joint_prob_p = cond_prob_p * prob_nearest_nei * 0.5
                        mutual_info += joint_prob_m * np.log(
                            cond_prob_m / y_nd.probs[0])
                        mutual_info += joint_prob_p * np.log(
                            cond_prob_p / y_nd.probs[1])
                y_nd.mutual_info = mutual_info

    def write_dot_file(self, fname):
        with open(fname, "w") as f:
            str0 = "digraph G {\n"
            for nd_id in range(1, NUM_DNODES+1):
                y_nd = self.y_nodes[nd_id-1]
                for nn  in y_nd.nearest_nei:
                    color = efficiency_to_hex(y_nd.get_efficiency())
                    str0 += (f'S{nn}->S{nd_id} [color="{color}"];\n')
                spin = y_nd.sample()
                #print("xdc", nd_id, ".", spin)
                if spin == -1:
                    str0 += (f"S{nd_id}[style=filled,fillcolor=black,"
                             f"fontcolor=white];\n")
            str0 += "}"
            f.write(str0)

if __name__ == "__main__":
    def main1():
        net = Net(beta=1, jj=.2, h=.3)
        for i in range(7):
            print("*******x_nodes[i]:")
            net.x_nodes[i].describe_self()
        print("---------------------")
        for i in range(7):
            print("*******y_nodes[i]:")
            net.y_nodes[i].describe_self()
    def main2(do_plot):
        net = Net(beta=1, jj=.2, h=.3)
        dot_file = "test.txt"
        net.write_dot_file(dot_file)
        if do_plot:
            plot_dot_with_colorbar(
                dot_file,
                cmap_name="viridis",
                vmin=0.0,
                vmax=1.0,
                colorbar_label="Efficiency"
            )
    #main1()
    main2(True)
