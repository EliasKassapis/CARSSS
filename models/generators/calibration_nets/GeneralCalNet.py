from models.generators.GeneralGenerator import GeneralGenerator


class GeneralCalNet(GeneralGenerator):

    def __init__(self, n_channels_in=(1), n_hidden=(1), n_channels_out=(1), device="cpu", **kwargs):  # CHECK DEFAULT VALUES!!!!!!!!
        super(GeneralCalNet, self).__init__(n_channels_in, device, **kwargs)

        self.n_channels_out = n_channels_out
        self.n_hidden = n_hidden

        #todo add temperature scaling, and logits return