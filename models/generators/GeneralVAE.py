import matplotlib
from models.generators.GeneralGenerator import GeneralGenerator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from utils.constants import *
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import _1hot_2_2d, _recolour_label, de_torch, move_color_channel


class GeneralVAE(GeneralGenerator):

    def __init__(self, n_channels_in=(1), n_hidden=(1), n_channels_out=(1), device="cpu", **kwargs):  # CHECK DEFAULT VALUES!!!!!!!!
        super(GeneralVAE, self).__init__(n_channels_in, device, **kwargs)

        self.n_channels_out = n_channels_out
        self.n_hidden = n_hidden

    def plot_sample_preds(self, images, labels, calnet_preds, pred_dist, gt_dist, n_preds, dataset):

        n_plotted_preds = 5 if n_preds > 5 else n_preds

        n_cols = n_plotted_preds + 4
        n_rows = pred_dist.shape[1]

        fig = plt.figure(figsize=(n_cols + 2, n_rows + 2))
        canvas = FigureCanvasAgg(fig)

        if dataset == "LIDC":
            lidc_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

        # plot sample predictions
        for i in range(n_rows):

            # plot input
            plt.subplot(n_rows, n_cols, i * n_cols + 1)
            plottable_images = move_color_channel(de_torch((images[i] + 1) / 2))
            if plottable_images.shape[-1] == 1: plottable_images = plottable_images.squeeze()
            plt.imshow(plottable_images, interpolation="none")
            if i == 0: plt.title("Input")
            plt.xticks([])
            plt.yticks([])

            for j in range(n_cols - 4):
                plottable_pred = _recolour_label(de_torch(_1hot_2_2d(pred_dist[j, i, :, :], sample=True)), dataset=dataset)

                plt.subplot(n_rows, n_cols, i * n_cols + j + 2)
                if dataset == "LIDC":
                    plt.imshow(plottable_pred, norm=lidc_norm, interpolation="none")
                else:
                    plt.imshow(plottable_pred, interpolation="none")
                if i == 0: plt.title(f"Pred {j + 1}")
                plt.xticks([])
                plt.yticks([])

            # plot average predictions
            plt.subplot(n_rows, n_cols, i * n_cols + n_cols - 2)
            plottable_avg_pred = _recolour_label(de_torch(_1hot_2_2d(pred_dist[:, i, :, :].mean(0), sample=True)), dataset=dataset)
            if dataset == "LIDC":
                plt.imshow(plottable_avg_pred, norm=lidc_norm, interpolation="none")
            else:
                plt.imshow(plottable_avg_pred, interpolation="none")
            if i == 0: plt.title(f"Avg Pred\nN = {pred_dist.shape[0]}")
            plt.xticks([])
            plt.yticks([])

            # plot calibration net predictions
            plt.subplot(n_rows, n_cols, i * n_cols + n_cols - 1)
            plottable_calnet_pred = _recolour_label(de_torch(_1hot_2_2d(calnet_preds[i], sample=True)), dataset=dataset)

            if dataset == "LIDC":
                plt.imshow(plottable_calnet_pred, norm=lidc_norm, interpolation="none")
            else:
                plt.imshow(plottable_calnet_pred, interpolation="none")
            if i == 0: plt.title("CalNet Pred")
            plt.xticks([])
            plt.yticks([])

            # plot actual predictions
            plt.subplot(n_rows, n_cols, i * n_cols + n_cols)

            if gt_dist is None:
                if labels.shape[1] != LABELS_CHANNELS:
                    label = torch.eye(LABELS_CHANNELS)[labels[:, 1, :, :].long()].permute(0, 3, 1, 2)[i] # convert rgb label to one-hot
                else:
                    label = labels[i]
                plottable_label = _recolour_label(de_torch(_1hot_2_2d(label,sample=True)), dataset=dataset)
            else:
                pad = lambda x: np.pad(x.cpu().numpy(), pad_width=2, mode='constant', constant_values=1)
                glued_top = np.concatenate((pad(gt_dist[i, 0]), pad(gt_dist[i, 1])), axis=1)
                glued_bottom = np.concatenate((pad(gt_dist[i, 2]), pad(gt_dist[i, 3])), axis=1)
                plottable_label = np.concatenate([glued_top, glued_bottom], axis=0)


            if dataset == "LIDC":
                plt.imshow(plottable_label, norm=lidc_norm, interpolation="none")
            else:
                plt.imshow(plottable_label, interpolation="none")
            if i == 0: plt.title("Label 0")
            plt.xticks([])
            plt.yticks([])

        fig.suptitle('Sample predictions')

        # convert figure to array
        canvas.draw()
        _, (width, height) = canvas.print_to_buffer()
        s = canvas.tostring_rgb()

        plt.close(fig)

        return np.fromstring(s, dtype='uint8').reshape((height, width, 3))

    @torch.no_grad()
    def plot_sample_figure(self, input_images, labels, calnet_preds, vae_input, n_samples, args):
        """
        Plot a grid of size n_samples * n_samples with sampled images
        """
        dataset = args.dataset

        pred_dist, log_p_score, avg_preds = self.sample(vae_input, n_samples)

        plotted_sample_preds = self.plot_sample_preds(input_images, labels, calnet_preds, pred_dist, n_preds=n_samples, dataset=dataset)

        return plotted_sample_preds
