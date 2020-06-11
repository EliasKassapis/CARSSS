from utils.constants import *
from utils.evaluation_utils import *
from utils.training_helpers import *
from utils.model_utils import save_models
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.generators.GeneralVAE import GeneralVAE
from models.generators.GeneralGenerator import GeneralGenerator
from utils.training_helpers import instance_checker
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def evaluation(dataloader_test, generator, calibration_net, discriminator, args, number_of_batches = 1, b_index=0, visualize = True, save=False, load = False, print_stats = True):

    if visualize:
        visualize_results(dataloader_test, generator, calibration_net, discriminator, args, number_of_batches=number_of_batches)
    else:
        avg_GED = []
        avgYS = []
        avgSS = []
        total_pixel_mode_counts = 0
        avg_calnet_class_probs = []

        for i, (batch) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            if i >= number_of_batches:
                break
            else:

                if args.dataset == "LIDC":
                    images, labels, gt_dist = unpack_batch(batch)
                    gt_labels = None
                else:
                    images, labels = unpack_batch(batch)
                    gt_dist = None
                    gt_labels = None

                if (args.dataset == "CITYSCAPES19" and args.class_flip):
                    gt_labels = labels.clone()
                    labels = torch.eye(LABELS_CHANNELS)[labels[:, 1, :, :].long()].permute(0, 3, 1, 2)
                    bb_preds = batch["bb_preds"].to(DEVICE).float()
                    bb_preds = torch.eye(LABELS_CHANNELS)[bb_preds[:, 1, :, :].long()].permute(0, 3, 1, 2).to(DEVICE)

                    overlapped_mask = get_cs_ignore_mask(bb_preds, labels) # get indexes of correct bb preds

                    unlabelled_idxs = torch.where(labels.argmax(1) == 24)  # get indexes of unlabelled pixels
                else:
                    bb_preds = None
                    overlapped_mask = None
                    unlabelled_idxs = None

                if args.mode == "test" and load:
                    images, labels, calnet_preds, pred_dist = load_numpy_arrays(i, args, to_tensor=True)
                    gt_labels = labels.argmax(1, keepdim=True)
                else:
                    calibration_net.eval()
                    generator.eval()
                    with torch.no_grad():

                        # forward pass
                        _, calnet_preds, calnet_labelled_imgs = calibration_net_forward_pass(calibration_net, images, bb_preds, unlabelled_idxs, args)

                        g_input = images if args.calibration_net == "EmptyCalNet" else calnet_labelled_imgs
                        pred_dist, _, _ = generator.sample(g_input, ign_idxs=unlabelled_idxs, n_samples=args.n_generator_samples_test)

                if save and not load:
                    if args.dataset == "LIDC":
                        s_labels = gt_dist
                    else:
                        s_labels = labels
                    save_numpy_arrays(images, s_labels, calnet_preds, pred_dist, batch_id=i, args=args)


                if args.dataset == "LIDC":
                    if not args.generator == "EmptyGenerator":
                        ged, _, d_YS, d_SS = compute_ged(pred_dist, gt_dist, calnet_preds, args=args, g_input = images, n_samples=args.n_generator_samples_test)
                        avg_GED.append(ged.mean())
                        avgYS.append(d_YS)
                        avgSS.append(d_SS)

                        print(f"\nGED_batch_{i} = {ged.mean().item()}")

                if args.dataset == "CITYSCAPES19" and args.class_flip:
                    if instance_checker(generator, GeneralVAE):
                        # get classes, corresponding flip classes and probabilities
                        flip_args = eval(f"{args.dataset}_{args.flip_experiment}FLIP")
                        class_1 = flip_args[0]
                        class_2 = flip_args[1]
                        flip_probs = flip_args[2]

                        label = gt_labels[:, 0, :, :]  # rgb channels are identical so we extract only one of them

                        calnet_class_probs = compute_pred_class_probs(labels, calnet_preds, overlapped_mask, args)
                        avg_calnet_class_probs.append(calnet_class_probs)

                        # get all modes
                        gt_dist = get_all_modes(label, input_classes = class_1, target_classes=class_2, flip_probs = flip_probs, n_flipped_modes = len(flip_probs))

                        ged, d_matrices, d_YS, d_SS = compute_ged(pred_dist, gt_dist, calnet_preds, args=args, n_samples=args.n_generator_samples_test)
                        avg_GED.append(ged.mean())

                        print(f"\nGED_batch_{i} = ", ged.mean().item())

                        avgYS.append(d_YS)
                        avgSS.append(d_SS)

                        pixel_mode_counts = count_pixel_modes(labels, pred_dist, input_classes = class_1, target_classes=class_2)
                        total_pixel_mode_counts += pixel_mode_counts.sum(0) #mean over the batch dim

        # mGED = torch.stack(avg_GED, dim=0).mean(0)
        mGED = nanmean(torch.stack(avg_GED, dim=0), dim=0)
        avgYS = torch.stack(avgYS, dim=0).mean(0)
        avgSS = torch.stack(avgSS, dim=0).mean(0)
        mYS = nanmean(avgYS)
        mSS = nanmean(avgSS)

        if (args.dataset == "CAMVID" or args.dataset == "CITYSCAPES19") and args.class_flip:

            f_classes_idxs = torch.stack((torch.LongTensor(class_1),torch.LongTensor(class_2)), dim=1)

            calnet_class_probs = torch.stack(avg_calnet_class_probs, dim=0).mean(0)[f_classes_idxs]

            normalized_pixel_mode_counts = total_pixel_mode_counts / total_pixel_mode_counts[:, 0].unsqueeze(1).expand(total_pixel_mode_counts.shape)
            pred_class_probs = normalized_pixel_mode_counts[:,1:] # 0 dim = total

            gt_mode_probs = get_mode_statistics(probabilities=flip_probs, n_flipped_modes=len(flip_probs))["probs"]

            # if save:
            save_results(gt_mode_probs, flip_probs, calnet_class_probs, pred_class_probs, mGED, mYS, mSS, args)

        # log stats
        if not args.debug and args.mode == "train":
            wandb.log({"meanGED": mGED.cpu()})

            wandb.log({"meanYS": mYS.cpu()})

            wandb.log({"meanSS": mSS.cpu()})


        if print_stats:
            print("\n---------------------------------------------")
            print("\nmGED = ", mGED)
            print("\nmYS = ", mYS)
            print("\nmSS = ", mSS)

            if args.dataset == "CITYSCAPES19" and args.class_flip:

                print("\n---------------------------------------------")
                print("\nGt class probs = ", flip_probs)
                print("\nCalibration Net class probs = ", calnet_class_probs)

                print("\nRefinement Net class probs = ", pred_class_probs)

                print("\n---------------------------------------------")


def visualize_results(dataloader, generator, calinration_net, discriminator, args, number_of_batches = 1):

    generator.eval()

    if isinstance(generator, GeneralVAE):
        if args.z_dim == 2:
            image = generator.plot_manifold(20, x1=0.1, x2=0.9)
            plt.imshow(image, interpolation="none")
            plt.show()
            plt.close()

    for i, (batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i >= number_of_batches:
            break

        if args.dataset == "LIDC":
            images, labels, gt_dist = unpack_batch(batch)
            gt_labels = None
        else:
            images, labels = unpack_batch(batch)
            gt_dist = None
            gt_labels = None

        if (args.dataset == "CITYSCAPES19" and args.class_flip):
            gt_labels = labels.clone()
            labels = torch.eye(LABELS_CHANNELS)[labels[:, 1, :, :].long()].permute(0, 3, 1, 2)
            bb_preds = batch["bb_preds"].to(DEVICE).float()
            bb_preds = torch.eye(LABELS_CHANNELS)[bb_preds[:, 1, :, :].long()].permute(0, 3, 1, 2).to(DEVICE)
        else:
            bb_preds = None

        calnet_preds, calnet_labelled_imgs, fake_labels, pred_dist, al_maps, gan_al_maps = test_forward_pass(images, labels, bb_preds, generator, calinration_net, discriminator, args)

        comparison_figure = plot_comparison_figure(batch, calnet_preds, fake_labels, al_maps, gan_al_maps, generator, calinration_net, discriminator, args)

        fig = plt.figure(1)
        ax = plt.gca()
        ax.axis('off')
        plt.imshow(comparison_figure, interpolation="none")
        plt.show()
        plt.close(fig)

        if instance_checker(generator, GeneralVAE):

            plotted_samples = generator.plot_sample_preds(images, labels, calnet_preds, pred_dist, gt_dist, n_preds=args.n_generator_samples_test, dataset=args.dataset)

            fig = plt.figure(3)
            ax = plt.gca()
            ax.axis('off')
            plt.imshow(plotted_samples, interpolation="none")
            plt.show()
            plt.close(fig)

def validation(validation_data, generator, calibration_net, discriminator, args, batch_idx):

    if args.dataset == "LIDC":
       plotting_batches = list(iter(validation_data))[4] # nice tumours
    else:
       plotting_batches = next(iter(validation_data))

    validation_plots(plotting_batches, generator, calibration_net, discriminator, args, batch_idx=batch_idx)

    evaluation(validation_data, generator, calibration_net, discriminator, args, b_index=batch_idx, number_of_batches=1, visualize=False, save=False, print_stats=True)

def validation_plots(batch,
                     generator,
                     calibration_net,
                     discriminator,
                     args,
                     batch_idx=0
                     ):

    assert instance_checker(generator, GeneralGenerator)
    assert instance_checker(calibration_net, GeneralGenerator)
    assert instance_checker(discriminator, GeneralDiscriminator)

    if args.dataset == "LIDC":
        images, labels, gt_dist = unpack_batch(batch)
        gt_labels = None
    else:
        images, labels = unpack_batch(batch)

        gt_dist = None
        gt_labels = None

    if args.dataset == "CITYSCAPES19":
        bb_preds = batch["bb_preds"].to(DEVICE).float()
        bb_preds = torch.eye(LABELS_CHANNELS)[bb_preds[:, 1, :, :].long()].permute(0, 3, 1, 2).to(DEVICE)

        one_hot_labels = torch.eye(LABELS_CHANNELS)[labels[:, 1, :, :].long()].permute(0, 3, 1, 2).to(DEVICE)
        overlapped_mask = get_cs_ignore_mask(bb_preds, one_hot_labels)
    else:
        bb_preds = None
        overlapped_mask = None

    if (args.dataset == "CITYSCAPES19" and args.class_flip):
        gt_labels = labels.clone()
        labels = torch.eye(LABELS_CHANNELS)[labels[:, 1, :, :].long()].permute(0, 3, 1, 2)

    calnet_preds, calnet_labelled_imgs, fake_labels, pred_dist, al_maps, gan_al_maps = test_forward_pass(images, labels, bb_preds, generator, calibration_net, discriminator, args)


    # save best calibration net
    if args.dataset == "LIDC":
        lab_dist = torch.eye(LABELS_CHANNELS)[(gt_dist).long()].permute(1,0,4,2,3).to(DEVICE).mean(0)
        eps=1e-7

        kl = lambda p, q: (-p.clamp(min=eps, max=1 - eps) * torch.log(q.clamp(min=eps, max=1 - eps))
                           + p.clamp(min=eps, max=1 - eps) * torch.log(p.clamp(min=eps, max=1 - eps))).sum(1)

        calnet_score = kl(calnet_preds.detach(), lab_dist).mean()

        if args.generator == "EmptyGenerator" and (not args.debug):
            wandb.log({"Calnet score": calnet_score})

        global BEST_CALNET_SCORE

        if args.mode == "train" and args.generator == "EmptyGenerator" and (not args.debug) and calnet_score is not None and calnet_score < BEST_CALNET_SCORE:
            BEST_CALNET_SCORE = calnet_score
            print(f"{PRINTCOLOR_GREEN} Saved New Best Calibration Net! {PRINTCOLOR_END}")
            save_models(discriminator, generator, calibration_net, f"Best_Model")

    # # log stats
    ged = compute_stats(args, generator, images, calnet_preds, calnet_labelled_imgs, fake_labels, pred_dist, gan_al_maps, labels, gt_dist, gt_labels, overlapped_mask, b_index=batch_idx)

    global BEST_GED

    if args.mode == "train" and (not args.debug) and ged is not None and ged < BEST_GED:
        BEST_GED = ged
        print(f"{PRINTCOLOR_GREEN} Saved New Best Model! {PRINTCOLOR_END}")
        save_models(discriminator, generator, calibration_net, f"Best_Model")

    # Plots
    comparison_figure = plot_comparison_figure(batch, calnet_preds, fake_labels, al_maps, gan_al_maps, generator, calibration_net, discriminator, args)

    if args.dataset == "CAMVID" or args.dataset == "CITYSCAPES19":
        calibration_figure = plot_calibration_figure(labels, calnet_preds, pred_dist, overlapped_mask, args)

    if instance_checker(generator, GeneralVAE):
        plotted_samples = generator.plot_sample_preds(images, labels, calnet_preds, pred_dist, gt_dist, n_preds=args.n_generator_samples_test, dataset=args.dataset)

    if not args.debug:
        # save and log

        comparison_figure = torch.from_numpy(np.moveaxis(comparison_figure, -1, 0)).float()
        save_example_images(comparison_figure, batch_idx, "comparison", "png")
        wandb.log({"Results": wandb.Image(vutils.make_grid(comparison_figure, normalize=True))})

        if args.dataset == "CITYSCAPES19":
            calibration_figure = torch.from_numpy(np.moveaxis(calibration_figure, -1, 0)).float()
            wandb.log({"Calibration": wandb.Image(vutils.make_grid(calibration_figure, normalize=True))})

        if instance_checker(generator, GeneralVAE):

            plotted_samples = torch.from_numpy(np.moveaxis(plotted_samples, -1, 0)).float()
            wandb.log({"Plotted samples": wandb.Image(vutils.make_grid(plotted_samples, normalize=True))})


@torch.no_grad()
def test_forward_pass(images,
                      labels,
                      bb_preds,
                      generator,
                      calibration_net,
                      discriminator,
                      args,
                      ):

    if args.dataset == "CITYSCAPES19":
        unlabelled_idxs = torch.where(labels.argmax(1) == 24)  # get indexes of unlabelled pixels
    else:
        unlabelled_idxs = None

    # check if we use a calibration net
    use_calibration_net = args.calibration_net != "EmptyCalNet"
    use_generator = args.generator != "EmptyGenerator"

    # ensure all models are in eval mode
    discriminator.eval()

    if use_calibration_net:
        calibration_net.eval()

        # forward pass calibration net
        _, calnet_preds, calnet_labelled_imgs = calibration_net_forward_pass(calibration_net, images, bb_preds, unlabelled_idxs, args)

        # get calibration pred aleatoric uncertainty maps
        al_maps = get_entropy(calnet_preds, CHANNEL_DIM)

    else:
        calnet_preds = None
        calnet_labelled_imgs = None
        al_maps = None

    if use_generator:
        generator.eval()

        # # forward pass generator
        g_input = images if args.calibration_net == "EmptyCalNet" else calnet_labelled_imgs
        pred_dist,_,_ = generator.sample(g_input, ign_idxs=unlabelled_idxs, n_samples=args.n_generator_samples_test)

        fake_labels = pred_dist[0]

        # get generator aleatoric uncertainty maps
        gan_al_maps = get_entropy(fake_labels, CHANNEL_DIM)

    else:
        fake_labels = None
        pred_dist = None
        gan_al_maps = None

    return calnet_preds, calnet_labelled_imgs, fake_labels, pred_dist, al_maps, gan_al_maps