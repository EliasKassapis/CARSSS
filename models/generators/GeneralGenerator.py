from models.GeneralModel import GeneralModel


class GeneralGenerator(GeneralModel):

    def __init__(self, n_channels_in=(1), n_hidden=(1), n_channels_out=(1), device="cpu", **kwargs):  # CHECK DEFAULT VALUES!!!!!!!!
        super(GeneralGenerator, self).__init__(n_channels_in, device, **kwargs)

        self.n_channels_out = n_channels_out
        self.n_hidden = n_hidden