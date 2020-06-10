from utils.training_helpers import instance_checker
import torch.nn as nn

class Trainer:

    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

    def prepare_training(self):
        for model in self.models:
            model.train()

    def do_backward(self, loss, G=False):
        for opt in self.optimizers:
            opt.zero_grad()

        if G:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        for opt in self.optimizers:
            opt.step()

    def grad_flow_check(self, model, loss, G=False):

        for opt in self.optimizers:
            opt.zero_grad()

        param_idx = [i for i, a in enumerate([*model.modules()]) if instance_checker(a, nn.Conv2d) or instance_checker(a, nn.ConvTranspose2d)][0]

        # store current param value
        sum_1 = list(model.modules())[param_idx].weight.sum().item()

        # perform backprop
        if G:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        for opt in self.optimizers:
            opt.step()

        sum_2 = list(model.modules())[param_idx].weight.sum().item()

        if (model.training):
            working = "working" if sum_1 != sum_2 else "\033[1;32m not working \033[0m"
            print("\n", model.__class__.__name__ + f" is in training mode, BACKPROP is {working} \n")
        else:
            working = "blocked" if sum_1 == sum_2 else "\033[1;32m not blocked \033[0m"
            print("\n",model.__class__.__name__ + f" is in eval mode, BACKPROP is {working} \n")

    def prepare_evaluation(self):
        """ sets models in evaluation mode """
        for model in self.models:
            model.eval()
