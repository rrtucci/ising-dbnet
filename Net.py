import itertools
from plotting import *
from math import prod

from Cond_Prob import *
from Node import *
from globals import *
from utils import *


class Net:
    def __init__(self, beta, jj, h=0, lam=0, num_iter=1, p0=.2,
                 do_reversing=False):
        self.beta = beta
        self.jj  = jj
        self.h = h
        self.lam = lam
        self.num_iter = num_iter
        self.p0 =p0
        self.cpt = Cond_Prob(beta, jj, h, lam)
        self.x_nodes = []
        self.y_nodes = []
        self.create_nodes(p0)
        for i in range(num_iter):
            if do_reversing:
                reversed_sweep = bool(i%2)
            else:
                reversed_sweep = False
            self.calc_y_node_params(reversed_sweep)
            self.mag = self.get_mag()
            self.av_eff = self.get_av_eff()
            if self.av_eff[1]:
                av_eff_str = f"{self.av_eff[0]:.5f}"
            else:
                av_eff_str = "undef"
            print(f"{i+1}, mag={self.mag:.5f}, av_eff={av_eff_str}")
            self.load_x_node_probs()

    def get_nd_from_id(self, id_num, type):
        assert id_num - 1 in range(NUM_DNODES)
        if type == "X":
            return self.x_nodes[id_num - 1]
        elif type == "Y":
            return self.y_nodes[id_num - 1]
        assert None, "this node type does not exist"

    def create_nodes(self, p0):
        for nd_id in range(1, NUM_DNODES + 1):
            x_node = Node(nd_id, "X", p0)
            y_node = Node(nd_id, "Y", p0)
            self.x_nodes.append(x_node)
            self.y_nodes.append(y_node)

    def calc_y_node_params(self, reversed_sweep=False):
        if not reversed_sweep:
            id_range = range(1,NUM_DNODES+1)
        else:
            id_range = reversed(range(1,NUM_DNODES+1))
        for nd_id in id_range:
            #print("lmjk", nd_id)
            y_nd = self.get_nd_from_id(nd_id, "Y")
            x_nd = self.get_nd_from_id(nd_id, "X")
            cond_info = 0
            prob_m = 0
            prob_p = 0
            num_nearest_nei = len(y_nd.nearest_nei)
            nearest_nei_x_nds = [self.get_nd_from_id(nd_id, "X") for
                                 nd_id in y_nd.nearest_nei]
            for nearest_nei_states in itertools.product(
                    [-1, 1], repeat=num_nearest_nei):
                prob_nearest_nei = prod(
                    [nearest_nei_x_nds[i].probs[
                         (nearest_nei_states[i] + 1) // 2] for i in \
                                         range(num_nearest_nei)])
                for x_spin in [-1, 1]:
                    x_nd_prob = x_nd.probs[(x_spin + 1) // 2]
                    cond_prob_m, cond_prob_p, zz = \
                        self.cpt.calc_cond_probs_y_if_abcd_x(
                            nearest_nei_states, x_spin)
                    joint_prob_m = (cond_prob_m * prob_nearest_nei *
                                    x_nd_prob)
                    joint_prob_p = (cond_prob_p * prob_nearest_nei *
                                    x_nd_prob)
                    prob_m += joint_prob_m
                    prob_p += joint_prob_p
                    # if nd_id == 12:
                    #     print("----------")
                    #     print("nearest, xspin", nearest_nei_states, x_spin)
                    #     print("mjrt-minus", cond_prob_m, prob_nearest_nei,
                    #           joint_prob_m, prob_m)
                    #     print("mjrt-plus", cond_prob_p, prob_nearest_nei,
                    #           joint_prob_p, prob_p)

                    cond_info -= joint_prob_m * np.log(cond_prob_m)
                    cond_info -= joint_prob_p * np.log(cond_prob_p)
            y_nd.probs = [prob_m, prob_p]
            y_nd.entropy = coin_toss_entropy(prob_m)
            y_nd.cond_info = cond_info
            y_nd.mutual_info = y_nd.entropy - cond_info
            y_nd.set_efficiency()
            # print("mgbyt-mutual, entropy, eff", y_nd.mutual_info,
            #       y_nd.entropy, y_nd.efficiency)

    def get_mag(self):
        mag = 0
        for y_nd in self.y_nodes:
            mag += -y_nd.probs[0] + y_nd.probs[1]
            # print("mnk", y_nd.probs[0], y_nd.probs[1], mag)
        return mag / NUM_DNODES

    def get_av_cond_info_and_entropy(self):
        sum_cond_info=0
        sum_ent = 0
        for y_nd in self.y_nodes:
            sum_cond_info += y_nd.cond_info
            sum_ent += y_nd.entropy
        return sum_ent/NUM_DNODES, sum_cond_info/NUM_DNODES

    def get_av_eff(self):
        sum_eff = 0
        num = 0
        no_undef_eff = True
        for y_nd in self.y_nodes:
            if y_nd.efficiency is not None:
                sum_eff += y_nd.efficiency
                num += 1
            else:
                no_undef_eff = False
        if num:
            av_eff = sum_eff/num
        else:
            av_eff = None
        return av_eff, no_undef_eff

    def load_x_node_probs(self):
        #print("mnk-----------------")
        for nd_id in range(1, NUM_DNODES + 1):
            y_nd = self.get_nd_from_id(nd_id, "Y")
            x_nd = self.get_nd_from_id(nd_id, "X")
            #print("llkxcvm" , x_nd.probs, y_nd.probs)
            x_nd.probs = y_nd.probs


    def write_dot_file(self, fname):
        with open(fname, "w") as f:
            str0 = "digraph G {\n"
            for nd_id in range(1, NUM_DNODES + 1):
                y_nd = self.y_nodes[nd_id - 1]
                for nn in y_nd.nearest_nei:
                    if y_nd.efficiency:
                        color = efficiency_to_hex(y_nd.efficiency)
                        label = f"{y_nd.efficiency:.2f}"
                        str0 += f'S{nn}->S{nd_id} [color="{color}",' \
                                f'label="{label}"];\n'
                    else:
                        str0 += f'S{nn}->S{nd_id} [style=dashed];\n'

                spin = y_nd.sample()
                # print("xdc", nd_id, ".", spin)
                if spin == -1:
                    str0 += (f"S{nd_id}[style=filled,fillcolor=black,"
                             f"fontcolor=white];\n")
            str0 += "}"
            f.write(str0)

    def plot_lattice(self, dot_file):
        self.write_dot_file(dot_file)
        if self.p0:
            p0_str = f"{self.p0:.3f}"
        else:
            p0_str = "random"
        if self.av_eff[1]:
            av_eff_str = f"{self.av_eff[0]:.3f}"
        else:
            av_eff_str = "undefined"
        caption = f"beta={self.beta:.3f}, jj={self.jj:.3f}, h={self.h:.3f}, " \
                  f"lam={self.lam:.3f}, num_iter={self.num_iter}, "\
                   f"p0={p0_str}, mag={self.mag:.3f}, av_eff={av_eff_str}"
        plot_dot_with_colorbar(dot_file, caption)

if __name__ == "__main__":
    def main1():
        net = Net(beta=1, jj=.2, h=.3, lam=.2)
        for i in range(7):
            print("*******x_nodes[i]:")
            net.x_nodes[i].describe_self()
        print("---------------------")
        for i in range(7):
            print("*******y_nodes[i]:")
            net.y_nodes[i].describe_self()

    def main2():
        beta_jj = BETA_JJ_CURIE * .5
        jj = 1  # no jj dependence when h=0
        beta = beta_jj / jj
        num_iter = 20
        net = Net(beta=beta,
                  jj=jj,
                  h=0,
                  lam=0,
                  num_iter=num_iter,
                  p0=None,
                  do_reversing=False) #no change if do reversing
        net.plot_lattice("test.txt")



    # main1()
    main2()