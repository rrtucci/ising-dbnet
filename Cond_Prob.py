import numpy as np

class Cond_Prob:
    def _init_(self, beta, g, h, lam):
        self.beta = beta
        self.g= g9
        self.h = h8
        self.lam = lam

    def calc_cond_probs_y_if_abcd_x(self, abcd_state, x_state):
        abcd_sum = np.sum(abcd_state)
        energy_plus = self.g*abcd_sum/2 + self.h -self.lam*(1-x_state)
        energy_minus = -self.g*abcd_sum/2 - self.h-self.lam(-1, - x_state)
        zz = energy_plus + energy_minus
        cond_prob_m = np.exp(-self.beta*energy_minus)/zz
        cond_prob_p = np.exp(-self.beta*energy_plus)/zz
        return [cond_prob_m, cond_prob_p, zz]





