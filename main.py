from data.Cityscapes19 import Cityscapes19
from data.LIDC import LIDC
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from utils.general_utils import *
from utils.model_utils import find_model, load_models_and_state
from utils.constants import *
from utils.data_utils import get_transforms_list
import argparse
from training.train import TrainingProcess
from testing.test import evaluation
import torchvision.transforms as transforms
import torch
import sys
import os
import wandb


def load_data(dataset_name, mode, batch_size, args):

    data = None

    if args.class_flip:
        assert args.dataset == "CITYSCAPES19" and LABELS_CHANNELS == 25, "Can only flip classes if CITYSCAPES with n_channles = 25"
    if LABELS_CHANNELS == 25:
        assert args.dataset == "CITYSCAPES19" and args.class_flip, "N_channels = 25 if CITYSCAPES19 with class flip is active"

    # compose transform object
    transforms_list = get_transforms_list(mode, args)
    transform = transforms.Compose(transforms_list)

    # initialize dataset
    if dataset_name == "CITYSCAPES19":
        dataset = Cityscapes19(mode, transform=transform)  # initialize dataset
    elif dataset_name == "LIDC":
        dataset = LIDC(mode, transform=transform)
    else:
        raise ValueError("Unexpected dataset name.")

    shuffle = mode == "train"

    if mode == "train" or mode == "val" or mode == "test":
        data = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=8)
    else:
        raise Exception(f"{mode} is not a valid dataset")

    print(f"finished loading {len(data)} batches for {mode} dataset")

    return data

def main(args):

    if not args.debug:
        # initialize logger
        wandb.init(config=args)

    if args.mode == "test":

        dataloader_test = load_data(args.dataset, "val" if args.dataset == "CITYSCAPES19" else "test", args.batch_size_plotting, args)

    else:
        dataloader_train = load_data(args.dataset, "train", args.batch_size, args)
        dataloader_validate = load_data(args.dataset, "val", args.batch_size_plotting, args)

    calibration_net = find_model(GEN_DIR, args.calibration_net,
                                 device=DEVICE,
                                 imsize=args.imsize,
                                 n_channels_in=INPUT_CHANNELS if args.dataset != "CITYSCAPES19" else INPUT_CHANNELS + LABELS_CHANNELS,
                                 n_channels_out=LABELS_CHANNELS,
                                 use_dropout=args.dropout,
                                 n_hidden=args.n_hidden_cal,
                                 temperature = args.temperature).to(DEVICE)

    generator = find_model(GEN_DIR, args.generator,
                           device=DEVICE,
                           imsize=args.imsize,
                           n_channels_in=INPUT_CHANNELS if args.calibration_net == "EmptyCalNet" else INPUT_CHANNELS + LABELS_CHANNELS,
                           n_channels_out=LABELS_CHANNELS,
                           z_dim = args.z_dim,
                           use_dropout=args.dropout,
                           n_hidden=args.n_hidden_gen).to(DEVICE)

    discriminator = find_model(DIS_DIR, args.discriminator,
                               device=DEVICE,
                               imsize=args.imsize,
                               n_channels_in= LABELS_CHANNELS if args.discriminator == "AEDiscriminator" else INPUT_CHANNELS + LABELS_CHANNELS,
                               use_dropout=args.dropout,
                               n_hidden=args.n_hidden_dis).to(DEVICE)

    # get models
    if args.pretrained:
        # load in state dicts
        load_models_and_state(discriminator,
                              generator,
                              calibration_net,
                              args.models_to_load,
                              args.pretrained_model_suffix,
                              args.pretrained_model_date)

    # assertions
    assert_type(GeneralGenerator, generator)
    assert_type(GeneralDiscriminator, discriminator)
    assert_type(GeneralGenerator, calibration_net)

    # train or test
    if (args.mode == "train"):

        # init optimizers
        gen_optimizer = torch.optim.Adam(params=generator.parameters(),lr=args.learning_rate_gen, betas=args.opt_betas, weight_decay=args.opt_weight_decay)
        dis_optimizer = torch.optim.Adam(params=discriminator.parameters(),lr=args.learning_rate_dis, betas=args.opt_betas, weight_decay=args.opt_weight_decay)
        cal_optimizer = torch.optim.Adam(params=calibration_net.parameters(),lr=args.learning_rate_cal, betas=args.opt_betas, weight_decay=args.opt_weight_decay)

        # init lr schedulers
        gen_lr_scheduler = MultiStepLR(gen_optimizer, milestones=args.lr_step_gen[0], gamma=args.lr_step_gen[1])
        dis_lr_scheduler = MultiStepLR(dis_optimizer, milestones=args.lr_step_dis[0], gamma=args.lr_step_dis[1])
        cal_lr_scheduler = MultiStepLR(cal_optimizer, milestones=args.lr_step_cal[0], gamma=args.lr_step_cal[1])


        # define loss functions
        if (not args.loss_gen == TOTAL_G_LOSS):
            print(f"{PRINTCOLOR_RED} WARNING: running with one generator-loss only: {args.loss_gen} {PRINTCOLOR_END}")
        weights_loss_functions = get_loss_weights(args)
        loss_gen = find_model(LOSS_DIR, TOTAL_G_LOSS, **weights_loss_functions)
        loss_dis = find_model(LOSS_DIR, TOTAL_D_LOSS, **weights_loss_functions)

        # assertions
        assert_type(GeneralLoss, loss_dis)
        assert_type(GeneralLoss, loss_gen)

        # define process
        train_progress = TrainingProcess(generator,
                                         discriminator,
                                         calibration_net,
                                         dataloader_train,
                                         dataloader_validate,
                                         gen_optimizer,
                                         dis_optimizer,
                                         cal_optimizer,
                                         gen_lr_scheduler,
                                         dis_lr_scheduler,
                                         cal_lr_scheduler,
                                         loss_gen,
                                         loss_dis,
                                         args)

        # train
        trained_succesfully = train_progress.train()

    elif (args.mode == "test"):

        # load in state dicts
        load_models_and_state(discriminator,
                              generator,
                              calibration_net,
                              args.test_models_to_load,
                              args.test_model_suffix,
                              args.test_model_date)

        # run test
        evaluation(dataloader_test, generator, calibration_net, discriminator, args, number_of_batches=20000, visualize=False, save=False, load = False)

    else:
        raise Exception(f"Unrecognized train/test mode?: {args.mode}")


def parse():
    parser = argparse.ArgumentParser()

    # Run mode ----------------------------------------------------------------------------------------------------------------
    parser.add_argument('--mode', default="train", type=str, help="'train' or 'test'")

    # debug
    parser.add_argument('--debug', default=True, type=str, help="If false, does not save directories")
    parser.add_argument('--timing', type=bool, default=False, help='are we measuring efficiency?')


    # Pre-training arguments --------------------------------------------------------------------------------------------------
    parser.add_argument('--pretrained', type=bool, default=False, help='Determines if we load a trained model or not')
    parser.add_argument('--resume', type=bool, default=False, help='Determines if we resume training on the pre-trained model or not')
    parser.add_argument('--pretrained_model_date', type=str, default="LIDC/Best_LIDC", help='date_stamp string for which model to load')
    parser.add_argument('--pretrained_model_suffix', type=str, default="Best_Model", help='filename string for which model to load')
    parser.add_argument('--models_to_load', type=list, default=["calibration_net"], help='list containing the name of models to load (["calibration_net", "generator", "discriminator"])')


    # Training arguments ------------------------------------------------------------------------------------------------------

    parser.add_argument('--epochs', default=1000000, type=int, help='max number of epochs')
    parser.add_argument('--max_training_minutes', type=int, default=7197, help='After which process is killed automatically')
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency (batch-wise) of evaluation')
    parser.add_argument('--plot_freq', type=int, default=400, help='Frequency (batch-wise) of plotting pictures')
    parser.add_argument('--saving_freq', type=int, default=50, help='Frequency (epoch-wise) of saving models')
    parser.add_argument('--device', default="cuda", type=str, help='device')

    parser.add_argument('--learning_rate_cal', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--learning_rate_gen', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--learning_rate_dis', type=float, default=1e-5, help='Learning rate')

    parser.add_argument('--lr_step_cal', type=float, default=([250,500], 1), help='Tuple enclosing list of epoch number where learning rate is changed and change factor')
    parser.add_argument('--lr_step_gen', type=float, default=([100,500,1000], 1), help='Tuple enclosing list of epoch number where learning rate is changed and change factor')
    parser.add_argument('--lr_step_dis', type=float, default=([30,500,1000], 1), help='Tuple enclosing list of epoch number where learning rate is changed and change factor')


    parser.add_argument('--opt_betas', type=tuple, default=(0.5,0.99), help='beta1 and beta2 for ADAM optimization')
    parser.add_argument('--opt_weight_decay', type=int, default=5e-4, help='weight decay for ADAM optimization')

    parser.add_argument('--schedule', type=tuple, default=(50,200), help='Alternate discriminator training for specified intervals - (update, freeze)')
    parser.add_argument('--DiscDAccCap', default=0.85, type=float, help="cap the discriminator accuracy at input value")


    # Test arguments -------------False-----------------------------------------------------------------------------------------------
    parser.add_argument('--test_model_date', default="LIDC/Best_LIDC", type=str, help='date_stamp string for which model to load')
    parser.add_argument('--test_model_suffix', default="Best_Model", type=str, help='filename string for which model to load')
    parser.add_argument('--test_models_to_load', type=list, default=["calibration_net", "generator"])

    # Model arguments -----------------------------------------------------------------------------------------------------------
    parser.add_argument('--calibration_net', default="SegNetCalNet", type=str, help="name of objectclass")
    parser.add_argument('--generator', default="UNetGenerator", type=str, help="name of objectclass")
    parser.add_argument('--discriminator', default="PixelDiscriminator", type=str, help="name of objectclass")

    parser.add_argument('--n_generator_samples_test', default=16, type=int, help="no. of samples to be used in plots and stats")

    parser.add_argument('--z_dim', default=8, type=int, help='dimensionality of latent code space')
    parser.add_argument('--n_hidden_cal', type=int, default=64, help='features in the first hidden layer')
    parser.add_argument('--n_hidden_gen', type=int, default=32, help='features in the first hidden layer')
    parser.add_argument('--n_hidden_dis', type=int, default=64, help='features in the first hidden layer')
    parser.add_argument('--temperature', type=torch.Tensor, default=torch.ones(LABELS_CHANNELS), help='specifies the magnitute of temperature scaling for the calibration net during test-time')
    parser.add_argument('--dropout', type=bool, default=False, help='specifies whether to use dropout')

    # MC dropout params
    parser.add_argument('--use_MC_dropout', type=bool, default=False, help='Specifies whether to use MC dropout')
    parser.add_argument('--n_MC_dropout_train', type=int, default=1, help='Specifies how many MC dropout forward passes to perform')
    parser.add_argument('--n_MC_dropout_test', type=int, default=100, help='Specifies how many MC dropout forward passes to perform')

    # Loss arguments -------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--loss_gen', default=TOTAL_G_LOSS, type=str, help="Overwrites hyperparams generatorloss if not total")
    parser.add_argument('--loss_dis', default=TOTAL_D_LOSS, type=str, help="name of objectclass")

    # hyperparams for CELoss  (-1 === DEFAULT)
    parser.add_argument('--CalNetLoss_weight', default=1, type=float, help="weight hyperparameter for calibration net loss")

    # hyperparams for Generatorloss
    parser.add_argument('--NonSaturatingGLoss_weight', default=10, type=float, help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--PixelLoss_weight', default=0, type=float, help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--CalLoss_weight', default=5, type=float, help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--ComplexityLoss_weight', default=0, type=float, help="weight hyperparameter for specific generatorloss")

    parser.add_argument('--n_cal_samples', default=5, type=float, help="Number of samples to use for prediction average in cal loss")

    # hyperparams for Discriminatorloss
    parser.add_argument('--DefaultDLoss_weight', default= 1, type=float, help="weight hyperparameter for specific discriminatorloss")
    parser.add_argument('--label_smoothing', default=False, type=bool, help="specifies whether to use label smoothing or not")

    # Data arguments -------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Size of batches loaded by the data loader.')
    parser.add_argument('--batch-size-plotting', type=int, default=5, help='Size of validation batch')
    parser.add_argument('--dataset', type=str, default='LIDC', help='LIDC, CITYSCAPES19 or CITYSCAPES35')
    parser.add_argument('--class_flip', type=bool, default=False, help="Specifies whether to randomly flip classes in CITYSCAPES")
    parser.add_argument('--flip_experiment', type = str, default = 'DEEP', help = "ROAD or DEEP") # flip only road or 5 classes flipped in the ProbabilistcUNet paper
    parser.add_argument('--crop', type=bool, default=True, help='Specifies whether to randomly crop dataset image or not')
    parser.add_argument('--resize', type=bool, default=True, help='Specifies whether to resize dataset image size or not')
    parser.add_argument('--noise', type=bool, default=False, help='Specifies whether to add random noise to labels')
    parser.add_argument('--imsize', type=tuple, default= (IMSIZE[0], IMSIZE[1]) if parser.get_default('resize') else eval(f"{parser.get_default('dataset')}_IMSIZE"),
                        help='Returns tuple of selected dataset image dimentions (specified in constants')

    assert len(parser.get_default('imsize')) == 2, f"Invalid IMSIZE values: {parser.get_default('imsize')}"

    return parser.parse_args()


if __name__ == '__main__':
    print("cuda_version:", torch.version.cuda, "pytorch version:", torch.__version__, "python version:", sys.version)
    print("Working directory: ", os.getcwd())
    args = parse()
    main(args)
