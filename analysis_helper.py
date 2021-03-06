import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from itertools import compress
from matplotlib import gridspec
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import svm
import random
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import LinearRegression
from linclab_utils import plot_utils

global figpath
# figpath = '' # path to save figure

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return my_autopct


def make_piechart(n_ramp_neurons, n_seq_neurons, n_neurons, figpath, label):
    neuron_counts = np.array([n_ramp_neurons, n_seq_neurons, (512 - n_neurons)])
    neuron_labels = ['Ramping cells', 'Sequence cells', 'Not included in analysis']
    plt.pie(neuron_counts, labels=neuron_labels, autopct=make_autopct(neuron_counts))
    plt.title(label)
    # plt.show()
    plt.savefig(figpath + '/neuron_counts.png')


def separate_ramp_and_seq(total_resp, norm=True):
    """
    Average the responses across episodes, normalize the activity according to the
    maximum and minimum of each cell (optional), and sort cells by when their maximum response happens.
    Then, Separate cells into ramping cells (strictly increasing/decreasing) and sequence cells.
    Note: sequence cells may contain NaN rows.
    - Arguments: total_resp, norm=True
    - Returns: cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp
    """
    np.seterr(divide='ignore', invalid='ignore')
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))  # len(segments) is also len_delay
    sorted_matrix = np.zeros((n_neurons, len(segments)))
    for i in range(len(segments)):  # at timestep i
        averages = np.mean(segments[i],
                           axis=0)  # 1 x n_neurons, each entry is the average response of this neuron at this time step across episodes
        unsorted_matrix[:, i] = np.transpose(
            averages)  # goes into the i-th column of unsorted_matrix, each row is one neuron
        if norm is True:
            normalized_matrix = (unsorted_matrix - np.min(unsorted_matrix, axis=1, keepdims=True)) / np.ptp(
                unsorted_matrix, axis=1, keepdims=True)
            # 0=minimum response of this neuron over time, 1=maximum response of this neuro over time
            max_indeces = np.argmax(normalized_matrix, axis=1)  # which time step does the maximum firing occur
            cell_nums = np.argsort(max_indeces)  # returns the order of cell number that should go into sorted_matrix
            for i, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[i] = normalized_matrix[i_cell]
        else:
            max_indeces = np.argmax(unsorted_matrix, axis=1)  # which time step does the maximum firing occur
            cell_nums = np.argsort(max_indeces)  # returns the order of cell number that should go into sorted_matrix
            for i, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[i] = unsorted_matrix[i_cell]
    # At this point, sorted_matrix should contain all cells
    assert len(sorted_matrix) == n_neurons

    ramp_up = np.all(sorted_matrix[:, 1:] >= sorted_matrix[:, :-1],
                     axis=1)  # Bool array with len=len(sorted_matrix). Want to remove True
    ramp_down = np.all(sorted_matrix[:, 1:] <= sorted_matrix[:, :-1],
                       axis=1)  # Bool array with len=len(sorted_matrix). Want to remove True
    # Want False in both ramp_up and ramp_down
    ramp = np.logical_or(ramp_up, ramp_down)  # Bool array
    seq = np.invert(ramp)  # Bool array
    cell_nums_seq, sorted_matrix_seq = cell_nums[seq], sorted_matrix[seq]
    cell_nums_ramp, sorted_matrix_ramp = cell_nums[ramp], sorted_matrix[ramp]
    return cell_nums, sorted_matrix, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp


def plot_sorted_averaged_resp(cell_nums, sorted_matrix, title, remove_nan=True):
    global figpath
    """
    Plot sorted normalized average-response matrix. On y-axis, display where in the layer the cell is.
    Note: normalize whole range, not just absolute value
    Arguments:
    - cell_nums
    - sorted_matrix
    - title: str
    """
    len_delay = np.shape(sorted_matrix)[1]
    entropy, ts, sqi = sequentiality_analysis(sorted_matrix)
    if remove_nan:
        # Remove NaNs
        mask = np.all(np.isnan(sorted_matrix), axis=1)
        sorted_matrix = sorted_matrix[~mask]
        cell_nums = cell_nums[~mask]
    fig, ax = plt.subplots(figsize=(6, 9))
    cax = ax.imshow(sorted_matrix, cmap='jet')
    cbar = plt.colorbar(cax, ax=ax, label='Normalized Unit Activation')
    ax.set_aspect('auto')
    ax.set_yticks([0, len(cell_nums)])
    ax.set_yticklabels(['1', f'{len(cell_nums)}'])
    ax.set_xticks(np.arange(len_delay, step=10))
    ax.set_xlabel('Time since delay onset')
    ax.set_ylabel('Unit #')
    ax.set_title(title + f' \n PE={entropy:.2f} \n TS={ts:.2f} \n SqI={sqi:.2f}')
    plt.show()
    #plt.savefig(figpath + f'/sorted_avg_resp_{title}.png')


def plot_sorted_in_same_order(resp_a, resp_b, a_title, b_title, big_title, len_delay, n_neurons, remove_nan=True):
    """
    Given response matrices a and b (plotted on the left and right, respectively),
    plot sorted_averaged_resp (between 0 and 1) for matrix a, then sort matrix b
    according the cell order that gives tiling pattern for matrix a
    Args:
    - resp_a, resp_b: arrays
    - a_title, b_title, big_title: strings
    - len_delay
    - n_neurons
    """
    global figpath
    segments_a = np.moveaxis(resp_a, 0, 1)
    segments_b = np.moveaxis(resp_b, 0, 1)
    unsorted_matrix_a = np.zeros((n_neurons, len_delay))
    unsorted_matrix_b = np.zeros((n_neurons, len_delay))
    sorted_matrix_a = np.zeros((n_neurons, len_delay))
    sorted_matrix_b = np.zeros((n_neurons, len_delay))
    for i in range(len(segments_a)):  # at timestep i
        averages_a = np.mean(segments_a[i],
                             axis=0)  # 1 x n_neurons, each entry is the average response of this neuron at this time step across episodes
        averages_b = np.mean(segments_b[i], axis=0)
        unsorted_matrix_a[:, i] = np.transpose(
            averages_a)  # goes into the i-th column of unsorted_matrix, each row is one neuron
        unsorted_matrix_b[:, i] = np.transpose(averages_b)
    normalized_matrix_a = (unsorted_matrix_a - np.min(unsorted_matrix_a, axis=1, keepdims=True)) / np.ptp(
        unsorted_matrix_a, axis=1, keepdims=True)
    # 0=minimum response of this neuron over time, 1=maximum response of this neuro over time
    normalized_matrix_b = (unsorted_matrix_b - np.min(unsorted_matrix_b, axis=1, keepdims=True)) / np.ptp(
        unsorted_matrix_b, axis=1, keepdims=True)
    max_indeces_a = np.argmax(normalized_matrix_a, axis=1)  # which time step does the maximum firing occur
    cell_nums_a = np.argsort(max_indeces_a)  # returns the order of cell number that should go into sorted_matrix
    for i, i_cell in enumerate(list(cell_nums_a)):
        sorted_matrix_a[i] = normalized_matrix_a[i_cell]
        sorted_matrix_b[i] = normalized_matrix_b[i_cell]  # sort b according to order in a

    if remove_nan:
        mask = np.logical_or(np.all(np.isnan(sorted_matrix_b), axis=1), np.all(np.isnan(sorted_matrix_b), axis=1))
        sorted_matrix_a = sorted_matrix_a[~mask]
        cell_nums_a = cell_nums_a[~mask]
        sorted_matrix_b = sorted_matrix_b[~mask]

    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(7, 8))
    ax.imshow(sorted_matrix_a, cmap='jet')
    ax2.imshow(sorted_matrix_b, cmap='jet')
    ax.set_ylabel("Unit #")
    ax.set_yticklabels(['1', f'{len(cell_nums_a)}'])
    ax2.set_yticklabels(['1', f'{len(cell_nums_a)}'])
    ax.set_title(a_title)
    ax2.set_title(b_title)
    ax.set_yticks([0, len(cell_nums_a)])
    ax2.set_yticks([0, len(cell_nums_a)])
    ax.set_xticks(np.arange(len_delay, step=10))
    ax2.set_xticks(np.arange(len_delay, step=10))
    ax.set_xlabel('Time since delay onset')
    ax2.set_xlabel('Time since delay onset')
    fig.suptitle(big_title)
    ax.set_aspect('auto')
    ax2.set_aspect('auto')
    # plt.show()
    plt.savefig(figpath + f'/sorted_in_same_order_{big_title}.png')



def split_train_and_test(percent_train, total_resp, total_stim, seed):
    """
    Split a neural activity matrix of shape n_stimuli x n_features into training
    (contains percent_train of data) and testing sets.
    Arguments:
    - percent_train (a number between 0 and 1)
    - total_resp (np.array of shape n_stimuli x n_features)
    - total_stim (np.array of shape n_stimuli x 1, each entry is 0 or 1)
    - seed
    Returns:
    - resp_train
    - resp_test
    - stimuli_train
    - stimuli_test
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_stimuli = total_resp.shape[0]
    n_train = int(percent_train * n_stimuli)  # use 60% of all data for training set
    ishuffle = torch.randperm(n_stimuli)
    itrain = ishuffle[:n_train]  # indices of data samples to include in training set
    itest = ishuffle[n_train:]  # indices of data samples to include in testing set
    stimuli_test = total_stim[itest]
    resp_test = total_resp[itest]
    stimuli_train = total_stim[itrain]
    resp_train = total_resp[itrain]
    return resp_train, resp_test, stimuli_train, stimuli_test


def decode_sample_from_single_time(total_resp, total_stim, n_fold=5):
    """
    Returns:
    - accuracies: array of shape n_fold x len_delay
    - accuracies_shuff: unit-shuffled. array of shape  n_fold x len_delay
    """
    from sklearn.model_selection import KFold
    len_delay = np.shape(total_resp)[1]
    accuracies = np.zeros((n_fold, len_delay))
    accuracies_shuff = np.zeros((n_fold, len_delay))
    kf = KFold(n_splits=n_fold)
    segments = np.moveaxis(total_resp, 0, 1)
    for t in range(len_delay): # for each time step
        resp = stats.zscore(segments[t], axis=1)  # z-normalized
        resp_shuff = np.stack([np.random.permutation(x) for x in resp])
        i_split = 0
        for train_index, test_index in kf.split(total_resp): # for each fold
            r_train, r_test = resp[train_index], resp[test_index]
            s_train, s_test = total_stim[train_index], total_stim[test_index]

            clf = svm.SVC()
            clf.fit(r_train, s_train)
            s_test_pred = clf.predict(r_test)
            accuracies[i_split, t] = np.mean(s_test_pred == s_test)

            r_train_shuff, r_test_shuff = resp_shuff[train_index], resp_shuff[test_index]
            s_train_shuff, s_test_shuff = total_stim[train_index], total_stim[test_index]
            clf = svm.SVC()
            clf.fit(r_train_shuff, s_train_shuff)
            s_test_pred_shuff = clf.predict(r_test_shuff)
            accuracies_shuff[i_split, t] = np.mean(s_test_pred_shuff == s_test_shuff)
            i_split += 1
    return accuracies, accuracies_shuff


def plot_decode_sample_from_single_time(total_resp, total_stim, title, n_fold=5, max_iter=100):
    """
    Arguments:
    - total_resp (eg. lstm, or first delay)
    - total_stim
    - title: str
    - max_iter: for LogisticRegression (default = 100)
    """
    global figpath
    accuracies, accuracies_shuff = decode_sample_from_single_time(total_resp, total_stim, n_fold=n_fold)
    len_delay = np.shape(total_resp)[1]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len_delay), np.mean(accuracies, axis=0), label='unshuffled', color=plot_utils.LINCLAB_COLS['green']) # TODO: green/purple for mem/nomem
    ax.fill_between(np.arange(len_delay), np.mean(accuracies, axis=0) - np.std(accuracies, axis=0), np.mean(accuracies, axis=0) + np.std(accuracies, axis=0), facecolor=plot_utils.LINCLAB_COLS['green'], alpha=0.5)
    ax.plot(np.arange(len_delay), np.mean(accuracies_shuff, axis=0), label='unit-shuffled', color=plot_utils.LINCLAB_COLS['grey'])
    ax.fill_between(np.arange(len_delay), np.mean(accuracies_shuff, axis=0) - np.std(accuracies_shuff, axis=0), np.mean(accuracies_shuff, axis=0) + np.std(accuracies_shuff, axis=0), facecolor=plot_utils.LINCLAB_COLS['grey'], alpha=0.5)
    ax.set(xlabel='Time since delay onset', ylabel='Stimulus decoding accuracy',
           title=title)
    ax.set_xticks(np.arange(len_delay, step=10))
    ax.legend()
    plt.show()
    #plt.savefig(figpath + f'/decode_stim_{title}.png')


def separate_vd_resp(resp, len_delay):
    '''
    :param resp: array of n_total_episodes x max(len_delays) x n_neurons
    :param len_delay: array of n_total_episodes
    :return: resp_dict: with keys = unique delay lengths, values = arrays storing resp corresponding to that delay length
    :return: counts: number of occurrence of each delay length
    '''
    len_delays, counts = np.unique(len_delay, return_counts=True)
    resp_dict = dict.fromkeys(len_delays)
    for ld in len_delays:  # for each unique delay length
        resp_dict[ld] = resp[len_delay == ld][:, :ld, :]
    return resp_dict, counts


def sequentiality_analysis(sorted_matrix):
    # Sequentiality index (Zhou 2020)
    # Arguments: sorted_matrix: n_neurons x len_delay
    # Return: peak entropy, temporal sparsity, sequentiality index
    len_delay = np.shape(sorted_matrix)[1]
    n_neurons = np.shape(sorted_matrix)[0]

    p_js = []  # to store p_j
    ts_ts = []  # to store TS(t)
    for t in range(len_delay):
        p_js.append(np.sum(np.argmax(sorted_matrix, axis=1) == t))  # number of units that peak at t
        r_i_t = sorted_matrix[:, t]
        r_i_t = r_i_t / np.nansum(r_i_t)
        ts_ts.append(np.nansum(-(r_i_t * np.log(r_i_t))) / np.log(n_neurons))  # TS(t)
    p_js = np.asarray(p_js) + 0.1  # add pseudocount to avoid log(0)
    ts_ts = np.asarray(ts_ts)
    peak_entropy = stats.entropy(p_js) / np.log(len_delay)
    temporal_sparsity = 1 - np.mean(ts_ts)

    sqi = np.sqrt(peak_entropy * temporal_sparsity)
    return peak_entropy, temporal_sparsity, sqi


def plot_sorted_vd(resp_dict, remove_nan=True):
    # Argument: resp_dict -- keys=length of delay, values = reps matrix
    # Sort each resp matrices according to the order of neurons of the first resp matrix, and plot as heat maps of n_neurons x len_delay
    # requirement: equal number of neurons in all resp matrices
    len_delays = list(resp_dict.keys())
    resp_matrices = list(resp_dict.values())
    resp_a = resp_matrices[0]
    len_delay = len_delays[0]
    n_neurons = np.shape(resp_a)[2]
    segments_a = np.moveaxis(resp_a, 0, 1)
    unsorted_matrix_a = np.zeros((n_neurons, len_delay))
    sorted_matrix_a = np.zeros((n_neurons, len_delay))
    sorted_matrices = []
    for i in range(len_delay):  # at timestep i
        averages_a = np.mean(segments_a[i], axis=0)
        unsorted_matrix_a[:, i] = np.transpose(averages_a)
    normalized_matrix_a = (unsorted_matrix_a - np.min(unsorted_matrix_a, axis=1, keepdims=True)) / np.ptp(
        unsorted_matrix_a, axis=1, keepdims=True)
    max_indeces_a = np.argmax(normalized_matrix_a, axis=1)
    cell_nums_a = np.argsort(max_indeces_a)  # Get the cell order
    for i, i_cell in enumerate(list(cell_nums_a)):
        sorted_matrix_a[i] = normalized_matrix_a[i_cell]
    sorted_matrices.append(sorted_matrix_a)

    for resp in resp_matrices[1:]:  # for the rest of the response matrices
        ld = np.shape(resp)[1]
        segments = np.moveaxis(resp, 0, 1)
        unsorted_matrix = np.zeros((n_neurons, ld))
        sorted_matrix = np.zeros((n_neurons, ld))
        for i in range(ld):  # at timestep i
            # 1 x n_neurons, each entry is the average response of this neuron at this time step across episodes
            averages = np.mean(segments[i], axis=0)
            # goes into the i-th column of unsorted_matrix, each row is one neuron
            unsorted_matrix[:, i] = np.transpose(averages)
        # 0=minimum response of this neuron over time, 1=maximum response of this neuro over time
        normalized_matrix = (unsorted_matrix - np.min(unsorted_matrix, axis=1, keepdims=True)) / np.ptp(unsorted_matrix,
                                                                                                        axis=1,
                                                                                                        keepdims=True)
        for i, i_cell in enumerate(list(cell_nums_a)):
            sorted_matrix[i] = normalized_matrix[i_cell]  # SORT ACCORDING TO RESP A'S ORDER
        sorted_matrices.append(sorted_matrix)

    if remove_nan:
        mask = np.logical_or(np.all(np.isnan(sorted_matrices[0]), axis=1), np.all(np.isnan(sorted_matrices[1]), axis=1), np.all(np.isnan(sorted_matrices[2]), axis=1))
        sorted_matrices[0] = sorted_matrices[0][~mask]
        sorted_matrices[1] = sorted_matrices[1][~mask]
        sorted_matrices[2] = sorted_matrices[2][~mask]

    from mpl_toolkits.axes_grid1 import AxesGrid
    fig = plt.figure(figsize=(6, 8))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, len(sorted_matrices)),
                    axes_pad=0.05,
                    share_all=False,
                    label_mode="L",
                    cbar_location="right",
                    cbar_size="20%",
                    cbar_mode="single",
                    )
    for sm, ax in zip(sorted_matrices, grid):
        print(len(sm))
        im = ax.imshow(sm, vmin=0, vmax=1, cmap='jet')
        ax.set_xticks(np.arange(10, np.shape(sm)[1] + 10, 10))
        ax.set_aspect(0.4)  # the smaller this number, the wider the plot. 1 means no horizontal stretch.
        ax.set_xlabel('Time')
        ax.set_ylabel("Unit #")
        ax.set_yticks([0, len(sm)])
        ax.set_yticklabels(['1', f'{len(sm)}'])
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
        cax.toggle_label(True)
    plt.show()


def time_decode(delay_resp, len_delay, n_neurons, bin_size, title, plot=False):
    """
    Decode time with multiclass logistic regression.
    :param delay_resp: n_episodes x len_delay x n_neurons
    :param len_delay: int
    :param n_neurons: int
    :param bin_size: int
    :param title: str
    :param plot: bool (default=False). Plot p_matrix as heatmap, with blue line indicating highest-probability decoded bin
    :return: p_matrix: len_delay (decoded) x len_delay (elapsed), each entry is probability of decoded time given resp at elapsed time
    :return: time_decode_error: mean absolute value of error-percentage
    :return: time_deocde_entropy: entropy of the probability matrix
    """
    global figpath
    p_matrix = np.zeros((len_delay, len_delay))
    clf = LogisticRegression(multi_class='multinomial')
    epi_t = np.array(np.meshgrid(np.arange(0, bin_size), np.arange(len_delay))).T.reshape(-1, 2)
    np.random.shuffle(epi_t)  # random combination of episode number and time
    percent_train = 0.6
    epi_t_train = epi_t[:int(percent_train * len_delay * bin_size)]  # 0.6*40000 by 2
    epi_t_test = epi_t[int(percent_train * len_delay * bin_size):]
    r_train = np.zeros((len(epi_t_train), n_neurons))
    r_test = np.zeros((len(epi_t_test), n_neurons))
    for i in range(len(epi_t_train)):
        r_train[i] = delay_resp[epi_t_train[i, 0], epi_t_train[i, 1], :]
    for i in range(len(epi_t_test)):
        r_test[i] = delay_resp[epi_t_test[i, 0], epi_t_test[i, 1], :]
    t_train = np.squeeze(epi_t_train[:, 1])
    t_test = np.squeeze(epi_t_test[:, 1])

    clf.fit(r_train, t_train)
    for t_elapsed in range(len_delay):
        p_matrix[:, t_elapsed] = np.mean(clf.predict_proba(r_test[t_test == t_elapsed]), axis=0)  # 1 x len_delay
    decoded_time = np.argmax(p_matrix, axis=0)
    # time_decode_rmsep = np.sqrt(np.mean(((decoded_time - np.arange(len_delay)) / len_delay)**2))  # root mean squared error-percentage
    time_decode_error = np.mean(
        np.abs((decoded_time - np.arange(len_delay)) / len_delay))  # mean absolute error-percentage
    time_decode_entropy = np.mean(stats.entropy(p_matrix, axis=0))

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(p_matrix, cmap='hot')
        ax.set_xlabel('Time since delay onset')
        ax.set_ylabel('Decoded time')
        cbar = plt.colorbar(cax, ax=ax, label='p', aspect=10, shrink=0.5)
        ax.set_title(f'Accuracy={100 * (1 - time_decode_error):.2f}%')
        ax.plot(np.arange(len_delay), decoded_time)
        ax.set_xticks([0, len_delay])
        ax.set_xticklabels(['0', str(len_delay)])
        ax.set_yticks([0, len_delay])
        ax.set_yticklabels(['0', str(len_delay)])
        plt.show()
        #plt.savefig(figpath + f'/decode_time_{title}.png')

    return p_matrix, time_decode_error, time_decode_entropy


def plot_dim_vs_delay_t(delay_resp, title, n_trials=5, var_explained=0.9):
    """
    Plot dimension (by default explains 90% variance) of single-time activities vs elapsed time.
    :param delay_resp: n_episodes x len_delay x n_neurons. Use a single analysis bin (eg. 1000 episodes) rather than
    entire training session.
    :param n_trials: number of trials to average over
    :param var_explained: how much variance you want the dimension. Default = 0.9
    """
    global figpath
    len_delay = np.shape(delay_resp)[1]
    dim = np.zeros((n_trials, len_delay-1))
    epi_shuff = np.arange(int(len(delay_resp)))
    np.random.shuffle(epi_shuff)
    for i_trial in range(n_trials):
        episode = epi_shuff[i_trial]
        for t in range(1, len_delay):
            delay_resp_t = delay_resp[episode, :t+1, :]
            pca_model = PCA()
            pca_model.fit(delay_resp_t)
            cumsum = pca_model.explained_variance_ratio_.cumsum()
            dim[i_trial, t-1] = next(x[0] for x in enumerate(cumsum) if x[1] > var_explained)
    fig, ax0 = plt.subplots()
    ax0.plot(np.arange(len_delay-1), np.mean(dim, axis=0), color=plot_utils.LINCLAB_COLS['blue'])
    ax0.fill_between(np.arange(len_delay-1), np.mean(dim, axis=0) - np.std(dim, axis=0), np.mean(dim, axis=0) + np.std(dim, axis=0), color='skyblue')
    ax0.set_xlabel('Time since delay onset')
    ax0.set_ylabel('Cumulative Dimensionality')
    ax0.set_title(title)
    #plt.show()
    plt.savefig(figpath + f'/dim_v_delay_{title}.png')


def single_cell_visualization(total_resp, binary_stim, cell_nums, type):
    global figpath
    len_delay = np.shape(total_resp)[1]
    n_neurons = np.shape(total_resp)[2]

    assert len(total_resp) == len(binary_stim)
    assert all(elem in list(np.arange(n_neurons)) for elem in cell_nums)

    for i_neuron in cell_nums:
        xl = total_resp[binary_stim == 0, :, i_neuron][-100:]
        xr = total_resp[binary_stim == 1, :, i_neuron][-100:]
        norm_xl = stats.zscore(xl, axis=1)
        norm_xr = stats.zscore(xr, axis=1)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(5, 8), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 2, 1.5]})
        fig.suptitle(f'Unit #{i_neuron}')

        im = ax1.imshow(norm_xl, cmap='jet')
        ax1.set_aspect('auto')
        ax1.set_xticks(np.arange(len_delay, step=10))
        ax1.set_yticks([0, len(norm_xl)])
        ax1.set_yticklabels(['1', '100'])
        ax1.set_ylabel(f'Left trials')

        im2 = ax2.imshow(norm_xr, cmap='jet')
        ax2.set_aspect('auto')
        ax2.set_xticks(np.arange(len_delay, step=10))
        ax2.set_yticks([0, len(norm_xr)])
        ax2.set_yticklabels(['1', '100'])
        ax2.set_ylabel(f'Right trials')

        ax3.plot(np.arange(len_delay), stats.zscore(np.mean(xl, axis=0), axis=0), label='Left', color=plot_utils.LINCLAB_COLS['yellow'])
        ax3.plot(np.arange(len_delay), stats.zscore(np.mean(xr, axis=0), axis=0), label='Right', color=plot_utils.LINCLAB_COLS['brown'])
        ax3.set_xlabel('Time since delay period onset')
        ax3.legend(loc='upper right', fontsize='medium')
        ax3.set_ylabel('Avg activation')
        # plt.show()
        plt.savefig(figpath + '/single_unit_v_time/' + type + f'/{i_neuron}.png')



