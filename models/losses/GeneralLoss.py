import torch.nn as nn


class GeneralLoss(nn.Module):

    def __init__(self, weight = 1.0):
        super(GeneralLoss, self).__init__()
        self.weight = weight
        self.active = (weight > 0)
        if (not self.active):
            print(f"{self.__class__.__name__} deactivated due to weight = {weight}")

    def forward(self, *input, **kwargs):
        """
        wrapper forward for other child class forward-methods so that weight can be applied

        :param input: any number of params
        :return:
        """

        # don't bother calculating if the loss is deactivated
        if (not self.active):
            return 0, {}

        loss = self.custom_forward(*input, **kwargs)

        output = self.weight * loss

        return output, {self.__class__.__name__: output.item()}

    def custom_forward(self, *input):
        """
        Method place-holder to be overridden in child-class


        :param input: any number of params
        :return:
        """
        raise Exception("PLEASE IMPLEMENT custom_forward METHOD IN CHILD-CLASS")