import numpy as np

class Arrow:

    def __init__(self,
                 pa1_id,
                 pa2_id,
                 probs_mm,
                 probs_pm,
                 probs_mp,
                 probs_pp):
        self.pa1_id = pa1_id
        self.pa2_id = pa2_id

    def calc_mutual_info(self):

