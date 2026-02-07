from Cond_Prob import *
from Node import *
from globals import *


class Net:
    def __init__(self, beta, g, h):
        self.cpt = Cond_Prob(beta, g, h)
        self.nodes = []
        self.arrows = []
        self.fill_nodes_and_arrows()

    def fill_nodes_and_arrows(self):
        nd_id = 1
        while nd_id <= NUM_TOP_ROOT_NODES:
            nd = Node(nd_id, "top_root")
            self.nodes.append(nd)
            nd_id += 1
        row = 2
        while row <= NUM_ROWS:
            while nd_id <= row*NUM_TOP_ROOT_NODES:
                if nd_id != row*NUM_TOP_ROOT_NODES:
                    nd = Node(nd_id, "nonroot")
                    pa1_nd = self.nodes[nd_id - NUM_TOP_ROOT_NODES +1]
                    pa2_nd = self.nodes[nd_id - NUM_TOP_ROOT_NODES + 2]
                    ar1 = Arrow(pa1_nd.nd_id,
                    probs_pp = self.cpt.probs_x_if_pp[0:2]*\
                        pa1_nd.probs_x[1] * pa2_nd.probs_x[1]
                    probs_pm = self.cpt.probs_x_if_pm[0:2] * \
                              pa1_nd.probs_x[1] * pa2_nd.probs_x[0]
                    probs_mp = self.cpt.probs_x_if_mp[0:2] * \
                              pa1_nd.probs_x[0] * pa2_nd.probs_x[1]
                    probs_mm = self.cpt.probs_x_if_mm[0:2] * \
                              pa1_nd.probs_x[0] * pa2_nd.probs_x[0]
                    nd.probs_x[0] = \
                        probs_pp[0] +probs_pm[0] + probs_mp[0] + probs_mm[0]
                    nd.probs_x[1] = \
                        probs_pp[1] + probs_pm[1] + probs_mp[1] + probs_mm[1]
                    self.nodes.append(nd)
                else:
                    nd = Node(nd_id, "side_root")
                    self.nodes.append(nd)
                nd_id += 1
            row += 1
