from models.losses.GeneralLoss import GeneralLoss
from utils.constants import *

class ComplexityLoss(GeneralLoss):

    def __init__(self, weight: float, **kwargs):
        super(ComplexityLoss, self).__init__(weight=weight)

    def custom_forward(self, mu, logvar, b_index, dataset_size):

        # compute closed-form KL divergence
        out = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        a = self.linear_annealing(0, 1, b_index, dataset_size)

        out = a*out

        return out


    def linear_annealing(self, start_weight, end_weight, b_index, annealing_steps):

        if annealing_steps == 0:
            return end_weight

        assert end_weight > start_weight
        current_weight = end_weight - start_weight
        annealed = min(start_weight + current_weight * b_index / annealing_steps, end_weight)

        return annealed