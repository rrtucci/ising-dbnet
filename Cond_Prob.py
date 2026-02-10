import numpy as np


class Cond_Prob:
    def __init__(self, beta, jj, h):
        self.beta = beta
        self.jj = jj
        self.h = h

    def calc_cond_probs_y_if_abcd_x(self, abcd_states, x_state):
        assert set(abcd_states).issubset(set([-1, 1])), \
            str(abcd_states) + " is illegal"
        assert x_state in [-1, 1]
        abcd_sum = np.sum(abcd_states)
        energy_plus = self.jj * abcd_sum / 2 + self.h
        energy_minus = -self.jj * abcd_sum / 2 - self.h
        # print("energy", energy_plus, energy_minus)
        energy_plus *= self.beta
        energy_minus *= self.beta
        if x_state != 1:
            cond_prob_p = 1e-4
        else:
            cond_prob_p = np.exp(-energy_plus)
        if x_state != -1:
            cond_prob_m = 1e-4
        else:
            cond_prob_m = np.exp(-energy_minus)
        # print("cp", cond_prob_m, cond_prob_p)
        zz = cond_prob_m + cond_prob_p
        cond_prob_m /= zz
        cond_prob_p /= zz
        return [cond_prob_m, cond_prob_p, zz]


if __name__ == "__main__":
    def main():
        cpt = Cond_Prob(beta=1.0, jj=.3, h=.2)
        print(cpt.calc_cond_probs_y_if_abcd_x([1, 1, -1, -1], -1))
    main()
