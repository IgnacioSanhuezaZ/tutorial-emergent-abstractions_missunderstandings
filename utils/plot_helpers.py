
import numpy as np
from matplotlib import pyplot as plt
import itertools

def plot_training_trajectory(results_train,
                             results_val,
                             message_length_train=None,
                             message_length_val=None,
                             steps=(1, 5),
                             figsize=(10, 7),
                             ylim=None,
                             xlim=None,
                             plot_indices=(1, 2, 3, 4, 5, 7),
                             plot_shape=(3, 3),
                             n_epochs=300,
                             train_only=False,
                             loss_plot=False,
                             message_length_plot=False,
                             titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)')):
    """ Plot the training trajectories for training and validation data"""
    plt.figure(figsize=figsize)

    for i, plot_idx in enumerate(plot_indices):
        plt.subplot(plot_shape[0], plot_shape[1], plot_idx)
        print(results_train)
        if message_length_plot:
            for j in range(len(message_length_train[i])):
                if j == 0:
                    n_epochs = len(message_length_train[i][j])
                else:
                    if len(message_length_train[i][j]) > n_epochs:
                        n_epochs = len(message_length_train[i][j])
            plt.plot(range(0, n_epochs, steps[0]), np.transpose(message_length_train[i]), color='green')
        else:
            plt.plot(range(0, n_epochs, steps[0]), np.transpose(results_train[i]), color='blue')
        if not train_only:
            plt.plot(range(0, n_epochs, steps[1]), np.transpose(results_val[i]), color='red')
            plt.legend(['train', 'val'])
            leg = plt.legend(['train', 'val'], fontsize=12)
            #leg.legendHandles[0].set_color('blue')
            #leg.legendHandles[1].set_color('red')
        plt.title(titles[i], fontsize=13)
        plt.xlabel('epoch', fontsize=12)
        if loss_plot:
            plt.ylabel('loss', fontsize=12)
        elif message_length_plot:
            plt.ylabel('message length', fontsize=12)
        else:
            plt.ylabel('accuracy', fontsize=12)
        if ylim:
            plt.ylim(ylim)
        if xlim:
            plt.xlim(xlim)

    # if loss_plot:
    #     plt.suptitle('loss', x=0.53, fontsize=15)
    # elif message_length_plot:
    #     plt.suptitle('message length', x=0.53, fontsize=15)
    # else:
    #     plt.suptitle('accuracy', x=0.53, fontsize=15)
    plt.tight_layout()
