import time
import wandb
from models.generators.calibration_nets.GeneralCalNet import GeneralCalNet
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from models.losses.CalNetLoss import CalNetLoss
from models.losses.TotalGeneratorLoss import TotalGeneratorLoss
from models.general.statistic import Statistic
from testing.test import validation_plots
from utils.general_utils import *
from models.general.trainer import Trainer
from utils.model_utils import save_models
from utils.training_helpers import *
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import sys
from torch import autograd


class TrainingProcess:

    def __init__(self,
                 generator,
                 discriminator,
                 calibration_net,
                 dataloader_train,
                 dataloader_validation,
                 optimizer_gen,
                 optimizer_dis,
                 optimizer_calnet,
                 gen_lr_sched,
                 dis_lr_sched,
                 calnet_lr_sched,
                 loss_gen,
                 loss_dis,
                 arguments):

        DATA_MANAGER.set_date_stamp()

        # models
        self.generator = generator
        self.discriminator = discriminator
        self.calibration_net = calibration_net

        # data
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation

        # optimizers
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis
        self.optimizer_calnet = optimizer_calnet

        # lr schedulers
        self.gen_lr_sched = gen_lr_sched
        self.dis_lr_sched = dis_lr_sched
        self.calnet_lr_sched = calnet_lr_sched

        # loss functions
        self.loss_gen = loss_gen
        self.loss_dis = loss_dis
        self.loss_calnet = CalNetLoss(arguments.CalNetLoss_weight)

        # save configuration
        self.args = arguments

        # trainers
        self.trainer_calnet = Trainer([calibration_net], [optimizer_calnet])
        self.trainer_gen = Trainer([generator], [optimizer_gen])
        self.trainer_dis = Trainer([discriminator], [optimizer_dis])

        # arguments
        self.args = arguments

        if self.args.dataset == "LIDC":
            self.plotting_batches = list(iter(self.dataloader_validation))[2]
        else:
            self.plotting_batches = list(iter(self.dataloader_validation))[8]

        # assert type
        assert_type(GeneralCalNet, calibration_net)
        assert_type(GeneralGenerator, generator)
        assert_type(GeneralGenerator, generator)
        assert_type(GeneralDiscriminator, discriminator)
        assert_type(GeneralLoss, loss_dis)
        assert_type(GeneralLoss, loss_gen)

        # assert nonzero
        assert_non_empty(arguments)
        assert_non_empty(optimizer_dis)
        assert_non_empty(optimizer_gen)
        assert_non_empty(dataloader_train)
        assert_non_empty(dataloader_validation)

        if arguments.dataset == "LIDC":
            assert LABELS_CHANNELS == 2, "Need to change LABELS_CHANNELS variable to: 2"
        elif arguments.dataset == "CITYSCAPES19":
            assert LABELS_CHANNELS == 25, "Need to change LABELS_CHANNELS variable to: 25"

    def batch_iteration(self,
                        batch,
                        b_index,
                        total_b_index,
                        train=True,
                        accuracy_discriminator=0,
                        ):
        """
         inner loop of epoch iteration

        """
        # unpack batch
        images, labels = unpack_batch(batch)

        if self.args.dataset == "CITYSCAPES19":
            unlabelled_idxs = torch.where(labels.argmax(1)==24) # get indices of unlabelled pixels
        else:
            unlabelled_idxs = None

        # combine x and y
        gt_labelled_imgs = torch.cat((images, labels), dim=CHANNEL_DIM)

        if train:
            # set all models to evaluation mode
            self.trainer_calnet.prepare_training()
            self.trainer_gen.prepare_training()
            self.trainer_dis.prepare_training()
        else:
            # set all models to evaluation mode
            self.trainer_calnet.prepare_evaluation()
            self.trainer_gen.prepare_evaluation()
            self.trainer_dis.prepare_evaluation()

        # if we are using a calibration net
        if self.args.calibration_net != "EmptyCalNet":

            if self.args.dataset == "CITYSCAPES19":
                bb_preds = batch["bb_preds"].to(DEVICE).float()
            else:
                bb_preds = None

            # forward pass calibration net
            if not (self.args.pretrained and "calibration_net" in self.args.models_to_load): # dont cache
                calnet_preds_logits, calnet_preds, calnet_labelled_imgs = calibration_net_forward_pass(self.calibration_net, images, bb_preds, unlabelled_idxs, self.args)

                # evaluate aletoric uncertainty
                with torch.no_grad():
                    aleatoric_maps = get_entropy(calnet_preds, dim=CHANNEL_DIM)

                # make sure that the weight for the calibration net loss is not zero!
                assert self.loss_calnet.active, "Calibration net loss is not active! (weight = 0)"

                # compute calibration net loss
                loss_calnet, loss_calnet_saving = self.loss_calnet(calnet_preds_logits, labels, self.args)

                # backward pass calibration net
                self.trainer_calnet.do_backward(loss_calnet)

            else:
                with torch.no_grad():
                    calnet_preds_logits, calnet_preds, calnet_labelled_imgs = calibration_net_forward_pass(self.calibration_net, images, bb_preds, unlabelled_idxs, self.args)

                    # evaluate aletoric uncertainty
                    aleatoric_maps = get_entropy(calnet_preds, dim=CHANNEL_DIM)

                    # compute calibration net loss
                    loss_calnet, loss_calnet_saving = self.loss_calnet(calnet_preds_logits, labels, self.args)  # todo add accountant weighting?

        else:
            calnet_preds_logits = None
            calnet_preds = None
            calnet_labelled_imgs = None
            aleatoric_maps = None
            loss_calnet = None
            loss_calnet_saving = {}

        # if we are using a generator
        if self.args.generator != "EmptyGenerator":

            # initialize variables
            preds = None
            pred_labelled = None
            loss_gen = None
            loss_gen_saving = {}


            # forward pass generator
            preds, pred_labelled, pred_dist, pred_dist_labelled = generator_forward_pass(self.generator, images, calnet_labelled_imgs, unlabelled_idxs, self.args)

            # compute generator lossmultinomial(p,1)
            loss_gen, loss_gen_saving =  self.loss_gen(images,
                                                       labels,
                                                       gt_labelled_imgs,
                                                       calnet_preds,
                                                       calnet_labelled_imgs,
                                                       preds,
                                                       pred_labelled,
                                                       pred_dist,
                                                       pred_dist_labelled,
                                                       self.generator,
                                                       self.discriminator,
                                                       self.args,
                                                       b_index,
                                                       len(self.dataloader_train)  # dataset size needed for scheduling of linear annealing
                                                       )


            if not (self.args.pretrained and "generator" in self.args.models_to_load) or self.args.resume: #todo add resume training argument?
                # backward pass generator
                self.trainer_gen.do_backward(loss_gen)
        else:
            preds = None
            pred_labelled = None
            loss_gen = None
            loss_gen_saving = {}

        # if we are using a discriminator
        if self.args.discriminator != "EmptyDiscriminator":

            # initialize variables
            # accuracy_discriminator = None
            loss_dis = None
            loss_dis_saving = {}

            if "D" in scheduler(total_b_index, *self.args.schedule):  # todo check if this works

                if accuracy_discriminator < self.args.DiscDAccCap:
                    self.optimizer_dis.zero_grad()

                # discriminator loss

                # real loss
                get_loss = torch.nn.BCELoss()

                gt_labelled_imgs.requires_grad_()
                real_scores = self.discriminator(gt_labelled_imgs)
                real_labels = torch.ones(real_scores.shape).to(DEVICE)
                real_loss = get_loss(real_scores, real_labels)

                if accuracy_discriminator < self.args.DiscDAccCap:
                    real_grad = autograd.grad(outputs=real_scores.sum(), inputs=gt_labelled_imgs, create_graph=True, retain_graph=True)[0]
                    r1 = torch.sum(real_grad.pow(2.0), dim=[1, 2, 3])

                # fake loss
                pred_labelled = pred_labelled.detach()
                fake_scores = self.discriminator(pred_labelled)
                fake_labels = torch.zeros(fake_scores.shape).to(DEVICE)
                fake_loss = get_loss(fake_scores, fake_labels)

                if accuracy_discriminator < self.args.DiscDAccCap:
                    total_dis_loss = (real_loss + fake_loss + 10*r1).mean()
                    total_dis_loss.backward()
                    self.optimizer_dis.step()

                loss_dis_saving = {"DefaultDLoss": (real_loss + fake_loss).item()}

                if accuracy_discriminator < self.args.DiscDAccCap:

                    loss_dis_saving = {**loss_dis_saving, "Regularized Loss": r1.mean().item()}

                accuracy_discriminator = compute_accuracy(torch.cat((real_scores, fake_scores), 0), torch.cat((real_labels, fake_labels), 0))

            else:
                with torch.no_grad(): # do not cache
                    # forward pass discriminator
                    combined_input, scores, gt_labels, accuracy_discriminator = discriminator_forward_pass(self.discriminator, gt_labelled_imgs, pred_labelled, self.args)

                    # compute dicsriminator loss
                    loss_dis, loss_dis_saving = self.loss_dis(self.discriminator, combined_input, scores, gt_labels, self.args)

        else:
            accuracy_discriminator = None
            loss_dis = None
            loss_dis_saving = {}

        # print flush
        sys.stdout.flush()

        return loss_calnet_saving, loss_gen_saving, loss_dis_saving, preds, aleatoric_maps, accuracy_discriminator

    def epoch_iteration(self, epoch_num):

        progress = []

        time_per_batch = []

        accuracy_discriminator = 0

        for i, (batch) in enumerate(self.dataloader_train):

            if self.args.timing:
                batch_start = time.process_time()

            # calculate amount of passed batch
            batches_passed = i + (epoch_num * len(self.dataloader_train))

            # run batch iteration
            loss_calnet, loss_gen, loss_dis, _, _, accuracy_discriminator = self.batch_iteration(batch, b_index=i, total_b_index = batches_passed, accuracy_discriminator=accuracy_discriminator)

            # assertions
            assert_type(dict, loss_gen)
            assert_type(dict, loss_dis)

            # print progress to terminal
            if (batches_passed % self.args.eval_freq == 0):
                # convert dicts to ints
                loss_gen_actual = sum(loss_gen.values())
                loss_dis_actual = sum(loss_dis.values())
                loss_calnet_actual = sum(loss_calnet.values()) if loss_calnet != None else loss_calnet

                # log to terminal and retrieve a statistics object
                statistic = self.log(loss_calnet_actual, loss_gen_actual, loss_dis_actual, loss_calnet, loss_gen,
                                     loss_dis, batches_passed,
                                     accuracy_discriminator)

                # assert type
                assert_type(Statistic, statistic)

                # append statistic to list
                progress.append(statistic)

                time_passed = datetime.now() - DATA_MANAGER.actual_date

                if (
                        (time_passed.total_seconds() > (
                                self.args.max_training_minutes * 60)) and self.args.max_training_minutes > 0):
                    raise KeyboardInterrupt(
                        f"Process killed because {self.args.max_training_minutes} minutes passed since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

            # save progress images and stats
            if (batches_passed % self.args.plot_freq == 0):
                validation_plots(self.plotting_batches, self.generator, self.calibration_net, self.discriminator, self.args, batch_idx=batches_passed)

            # empty cache
            torch.cuda.empty_cache()

            if self.args.timing:
                batch_end = time.process_time() - batch_start
                time_per_batch.append(batch_end)

        return progress, time_per_batch

    def log(self, loss_calnet, loss_gen, loss_dis, loss_calnet_dict, loss_gen_dict, loss_dis_dict, batches_passed, disc_accuracy):

        # put models in evaluation mode
        self.trainer_dis.prepare_evaluation()
        self.trainer_gen.prepare_evaluation()
        self.trainer_calnet.prepare_evaluation()

        # pass stats to wandb
        if self.args.debug == False:
            for e in list(loss_gen_dict.keys()):
                wandb.log({f'loss/gen/{e}': loss_gen_dict[e]})
            for e in list(loss_dis_dict.keys()):
                wandb.log({f'loss/dis/{e}': loss_dis_dict[e]})

            wandb.log({"loss/gen/total": loss_gen})
            if not self.args.discriminator == "EmptyDiscriminator":
                wandb.log({"loss/dis/total": loss_dis})
                wandb.log({"accuracy/dis":disc_accuracy})

            if loss_calnet != None:
                for e in list(loss_calnet_dict.keys()):
                    wandb.log({f'loss/calnet/{e}': loss_calnet_dict[e]})

        # validate on validationset
        loss_gen_validate, loss_dis_validate, loss_calnet_validate = 0, 0, 0

        stat = Statistic(loss_calnet_train=loss_calnet,
                         loss_gen_train=loss_gen,
                         loss_dis_train=loss_dis,
                         loss_calnet_val=loss_calnet_validate,
                         loss_gen_val=loss_gen_validate,
                         loss_dis_val=loss_dis_validate,
                         loss_calnet_train_dict=loss_calnet_dict,
                         loss_gen_train_dict=loss_gen_dict,
                         loss_dis_train_dict=loss_dis_dict,
                         dis_acc=disc_accuracy)

        # print
        print(
            f"",
            f"batch: {batches_passed}/{len(self.dataloader_train)}",
            f"|\t {stat}",
            f"details: Gen = {loss_gen_dict}, Dis = {loss_dis_dict}")

        return stat

    def train(self):

        # setup data output directories:
        if self.args.debug:
            print("\n \033[1;32m Note: DEBUG mode active!!!! \033[0m \n")
        else:
            setup_directories()

        # data gathering
        progress = []

        try:

            print(f"{PRINTCOLOR_BOLD}Started training with the following config:{PRINTCOLOR_END}\n{self.args}")

            time_per_epoch = []
            avg_time_per_batch = []

            # run
            for epoch in range(self.args.epochs):

                if self.args.timing:
                    epoch_start = time.process_time()

                print(
                    f"\n\n{PRINTCOLOR_BOLD}Starting epoch{PRINTCOLOR_END} {epoch}/{self.args.epochs} at {str(datetime.now())}")

                # do epoch
                epoch_progress, time_per_batch = self.epoch_iteration(epoch)

                # update learning rate
                self.gen_lr_sched.step()
                self.dis_lr_sched.step()
                self.calnet_lr_sched.step()

                # add progress
                progress += epoch_progress

                if self.args.debug == False:
                    # write models if needed (don't save the first one
                    if (((epoch + 1) % self.args.saving_freq) == 0):
                        save_models(self.discriminator, self.generator, self.calibration_net,
                                    f"Models_at_epoch_{epoch}")

                # flush prints
                sys.stdout.flush()

                if self.args.timing:
                    epoch_end = time.process_time() - epoch_start
                    time_per_epoch.append(epoch_end)
                    avg_time_per_batch.append(np.mean(time_per_batch))

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
            if self.args.debug == False: save_models(self.discriminator, self.generator, self.calibration_net, f"KILLED_at_epoch_{epoch}")
            return False
        except Exception as e:
            print(e)
            if self.args.debug == False: save_models(self.discriminator, self.generator, self.calibration_net, f"CRASH_at_epoch_{epoch}")
            raise e

        # flush prints
        sys.stdout.flush()

        if self.args.debug == False:
            save_models(self.discriminator, self.generator, self.calibration_net, "finished")

        return True
