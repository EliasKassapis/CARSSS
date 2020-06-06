
"""
Code adapted from https://github.com/StijnVerdenius/DeepFaceImageSynthesis
"""

class Statistic:
    def __init__(self, loss_calnet_train=None,
                 loss_gen_train=None,
                 loss_dis_train=None,
                 loss_calnet_val=None,
                 loss_gen_val=None,
                 loss_dis_val=None,
                 loss_calnet_train_dict=None,
                 loss_gen_train_dict=None,
                 loss_dis_train_dict=None,
                 loss_calnet_val_dict=None,
                 loss_gen_val_dict=None,
                 loss_dis_val_dict=None,
                 dis_acc=None):
        # floats
        self.loss_calnet_train = loss_calnet_train
        self.loss_gen_train = loss_gen_train
        self.loss_dis_train = loss_dis_train
        self.loss_calnet_val = loss_calnet_val
        self.loss_gen_val = loss_gen_val
        self.loss_dis_val = loss_dis_val
        self.dis_acc = dis_acc

        # dictionaries
        self.loss_calnet_train_dict = loss_calnet_val_dict
        self.loss_gen_train_dict = loss_gen_train_dict
        self.loss_dis_train_dict = loss_dis_train_dict

        self.loss_calnet_val_dict = loss_calnet_val_dict
        self.loss_gen_val_dict = loss_gen_val_dict  # not implemented
        self.loss_dis_val_dict = loss_dis_val_dict  # not implemented

    def __repr__(self):

        if self.loss_calnet_train == None:

            return f"loss-generator-train: {self.loss_gen_train :0.3f}, " + \
                   f"loss-discriminator-train: {self.loss_dis_train:0.3f}, " + \
                   f"accuracy-discriminator: {self.dis_acc} "

        else:

            return f"loss-calnet-train: {self.loss_calnet_train :0.3f}, " + \
                   f"loss-generator-train: {self.loss_gen_train :0.3f}, " + \
                   f"loss-discriminator-train: {self.loss_dis_train:0.3f}, " + \
                   f"accuracy-discriminator: {self.dis_acc} "


    def __str__(self):
        return self.__repr__()
