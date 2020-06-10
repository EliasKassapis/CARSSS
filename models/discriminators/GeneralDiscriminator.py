from models.GeneralModel import GeneralModel


class GeneralDiscriminator(GeneralModel):

    def __init__(self, n_channels_in: int, device:str="cpu", **kwargs):
        super(GeneralDiscriminator,self).__init__(n_channels_in, device, **kwargs)