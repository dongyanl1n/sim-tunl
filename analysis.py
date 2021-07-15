from analysis_helper import *
from world import *
import plot_utils

plot_utils.linclab_plt_defaults()
plot_utils.set_font(font='Helvetica')

data = np.load('/Users/dongyanlin/Desktop/manuscript_data/2021_06_11_23_51_15_tunl_nomem.npz')
label = 'NoMem NLF'
global figpath
figpath = '/Users/dongyanlin/Desktop/manuscript_figures/NoMem_NLF'

stim = data['stim']  # n_episode x 2
delay_resp_hx = data['delay_resp_hx']  # n_episode x len_delay x n_neurons
delay_resp_cx = data['delay_resp_cx']  # n_episode x len_delay x n_neurons
delay_loc = data['delay_loc']  # n_episode x len_delay x 2
choice = data['choice']  # n_episode x 2
action = data['action']  # n_episode x len_delay

# Select units with large enough variation in its activation
big_var_neurons = []
for i_neuron in range(512):
    if np.ptp(np.concatenate(delay_resp_hx[:, :, i_neuron])) > 0.0000001:
        big_var_neurons.append(i_neuron)

delay_resp = delay_resp_hx[:, 2:, [x for x in range(512) if x in big_var_neurons]]
delay_loc = delay_loc[:, 2:, :]

n_episodes = np.shape(delay_resp)[0]
len_delay = np.shape(delay_resp)[1]
n_neurons = np.shape(delay_resp)[2]

# separate left and right trials
left_stim_resp = delay_resp[np.all(stim == [1, 1], axis=1)]
right_stim_resp = delay_resp[np.any(stim != [1, 1], axis=1)]
left_stim_loc = delay_loc[np.all(stim == [1, 1], axis=1)]  # delay locations on stim==left trials
right_stim_loc = delay_loc[np.any(stim != [1, 1], axis=1)]

left_choice_resp = delay_resp[np.all(choice == [1, 1], axis=1)]
right_choice_resp = delay_resp[np.any(choice != [1, 1], axis=1)]
left_choice_loc = delay_loc[np.all(choice == [1, 1], axis=1)]  # delay locations on first_choice=left trials
right_choice_loc = delay_loc[np.any(choice != [1, 1], axis=1)]

binary_stim = np.ones(np.shape(stim)[0])
binary_stim[np.all(stim == [1, 1], axis=1)] = 0  # 0 is L, 1 is right

binary_choice = np.ones(np.shape(choice)[0])
binary_choice[np.all(choice == [1, 1], axis=1)] = 0  # 0 is L, 1 is right

binary_nonmatch = np.ones(np.shape(stim)[0])
binary_nonmatch[binary_stim == binary_choice] = 0  # 0 is match, 1 is nonmatch

correct_resp = delay_resp[binary_nonmatch == 1]
incorrect_resp = delay_resp[binary_nonmatch == 0]
correct_loc = delay_loc[binary_nonmatch == 1]  # delay locations on correct trials
incorrect_loc = delay_loc[binary_nonmatch == 0]

cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(
    delay_resp, norm=True)
delay_resp_ramp = delay_resp[:, :, cell_nums_ramp]
delay_resp_seq = delay_resp[:, :, cell_nums_seq]
left_stim_resp_ramp = delay_resp_ramp[np.all(stim == [1, 1], axis=1)]
right_stim_resp_ramp = delay_resp_ramp[np.any(stim != [1, 1], axis=1)]
left_stim_resp_seq = delay_resp_seq[np.all(stim == [1, 1], axis=1)]
right_stim_resp_seq = delay_resp_seq[np.any(stim != [1, 1], axis=1)]
n_ramp_neurons = len(cell_nums_ramp)
n_seq_neurons = len(cell_nums_seq)

# print('Make a pie chart of neuron counts...')
# make_piechart(n_ramp_neurons, n_seq_neurons, n_neurons, figpath, label)


# print('Sort avg resp analysis...')
#plot_sorted_averaged_resp(cell_nums_seq, sorted_matrix_seq, title=label+' Sequence cells', remove_nan=True)
#plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=label+' Ramping cells', remove_nan=True)
#plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=label+' All cells', remove_nan=True)

# print('dim v delay t analysis...')
#plot_dim_vs_delay_t(left_stim_resp, title=label+' All cells (Left trials)', n_trials=10, var_explained=0.99)
#plot_dim_vs_delay_t(left_stim_resp_ramp, title=label+' Ramping cells (Left trials)', n_trials=10, var_explained=0.99)
#plot_dim_vs_delay_t(left_stim_resp_seq, title=label+' Sequence cells (Left trials)', n_trials=10,  var_explained=0.99)
#plot_dim_vs_delay_t(right_stim_resp, title=label+' All cells (Right trials)', n_trials=10, var_explained=0.99)
#plot_dim_vs_delay_t(right_stim_resp_ramp, title=label+' Ramping cells (Right trials)', n_trials=10, var_explained=0.99)
#plot_dim_vs_delay_t(right_stim_resp_seq, title=label+' Sequence cells (Right trials)', n_trials=10,  var_explained=0.99)

# print('sort in same order analysis...')
# plot_sorted_in_same_order(left_stim_resp_ramp, right_stim_resp_ramp, 'Left', 'Right', big_title=label+' Ramping cells', len_delay=len_delay, n_neurons=n_ramp_neurons)
# plot_sorted_in_same_order(left_stim_resp_seq, right_stim_resp_seq, 'Left', 'Right', big_title=label+' Sequence cells', len_delay=len_delay, n_neurons=n_seq_neurons)
# plot_sorted_in_same_order(left_stim_resp, right_stim_resp, 'Left', 'Right', big_title=label+' All cells', len_delay=len_delay, n_neurons=n_neurons)
# plot_sorted_in_same_order(correct_resp, incorrect_resp, 'Correct', 'Incorrect', big_title=label+' All cells ic', len_delay=len_delay, n_neurons=n_neurons)

print('decode stim analysis...')
#plot_decode_sample_from_single_time(delay_resp, binary_stim, label+' All Cells', n_fold=5, max_iter=100)
#plot_decode_sample_from_single_time(delay_resp_ramp, binary_stim, label+' Ramping Cells', n_fold=5, max_iter=100)
plot_decode_sample_from_single_time(delay_resp_seq, binary_stim, label+' Sequence Cells', n_fold=7, max_iter=100)

# print('decode time analysis...')
# time_decode(delay_resp, len_delay, n_neurons, 1000, title=label+' All cells', plot=True)
# time_decode(delay_resp_ramp, len_delay, n_ramp_neurons, 1000, title=label+' Ramping cells', plot=True)
# time_decode(delay_resp_seq, len_delay, n_seq_neurons, 1000, title=label+' Sequence cells', plot=True)

#print('Single-cell visualization...')
#single_cell_visualization(delay_resp, binary_stim, cell_nums_ramp, type='ramp')
#single_cell_visualization(delay_resp, binary_stim, cell_nums_seq, type='seq')

# print('pca...')
#pca_analysis(delay_resp, binary_stim)

#print('single cell ratemaps...')
#plot_LvR_ratemaps(delay_resp, delay_loc, left_stim_resp, left_stim_loc, right_stim_resp, right_stim_loc, cell_nums_all,cell_nums_seq, cell_nums_ramp, label)

print('Analysis finished')
'''
from statsmodels.formula.api import glm
import statsmodels.api as sm
import pandas as pd
from scipy.stats.distributions import chi2


def likelihood_ratio(llmin, llmax):
    """
    Arguments:
    - llmin: llf for the nested model (with a covariate removed)
    - llmax: llf for the full model
    Return: Deviance of nested model from full model
    """
    return (2 * (llmax - llmin))


def pseudo_r2(model, sat_model):
    """
    Measures the degree to which a nested model captures the variance in firing rate. (Kraus et al., 2015)
    Arguments:
    - model: glm model
    Return:
    - pr2: Between 0 (nested model is no better than a null model) and 1 (nested model is as good as saturated model).
    """
    #return (model.llf - model.llnull) / (sat_model.llf - model.llnull)
    return 1- model.deviance / model.null_deviance


s_sig_neurons = []
t_sig_neurons = []
a_sig_neurons = []
p_sig_neurons = []

cells = cell_nums_all
dev = np.zeros((len(cells), 4))
pr2 = np.zeros((len(cells), 5))

mod_sig_threshold = 0.001
for i_neuron in cells[:3]:
    y = delay_resp[:, :, i_neuron].flatten()  # trial1 trial2 trial3..
    stim_ = np.repeat(binary_stim, len_delay)  # 000001111100000.... # stimulus
    time = np.tile(np.arange(len_delay), n_episodes)  # 012340123401234... # time
    act = action[:, 1:39].flatten()  # previous action
    locx = delay_loc[:, :, 1].flatten()
    locy = delay_loc[:, :, 0].flatten()
    arr = np.vstack((y, stim_, time, act, locx, locy)).T
    dta = pd.DataFrame(arr, columns=['y', 'stim', 'time', 'act', 'locx', 'locy'])
    family = sm.families.Gaussian()

    one_hot_cols = []
    one_hot_formula = 'y ~ stim + time + act + locx + locy'
    for i in range(38):
        one_hot_cols.append('t'+str(i))
        one_hot_formula += ' + t'+str(i)
    one_hot_time = np.zeros((time.size, time.max() + 1))  # 38000 x 38
    one_hot_time[np.arange(time.size), time] = 1
    one_hot_time = pd.DataFrame(one_hot_time, columns=one_hot_cols)
    data_sat = pd.concat([dta[['stim', 'act', 'locx', 'locy']], one_hot_time], axis=1)
    #print(one_hot_formula)
    #print('dta: ', dta)
    #print('data_sat: ', data_sat)


    model_sta = glm(formula='y ~ stim + time + act', data=dta, family=family).fit()
    #model_sta = sm.GLM(endog=dta['y'], exog=dta[['stim', 'time', 'act']], family=family).fit()
    model_stp = glm(formula='y ~ stim + time + locx + locy', data=dta, family=family).fit()
    #model_stp = sm.GLM(endog=dta['y'], exog=dta[['stim', 'time', 'locx', 'locy']], family=family).fit()
    model_sap = glm(formula='y ~ stim + act + locx + locy', data=dta, family=family).fit()
    #model_sap = sm.GLM(endog=dta['y'], exog=dta[['stim', 'act', 'locx', 'locy']], family=family).fit()
    model_tap = glm(formula='y ~ time + act + locx + locy', data=dta, family=family).fit()
    #model_tap = sm.GLM(endog=dta['y'], exog=dta[['time', 'act', 'locx', 'locy']], family=family).fit()
    model_full = glm(formula='y ~ stim + time + act + locx + locy', data=dta, family=family).fit()
    #model_full = sm.GLM(endog=dta['y'], exog=dta[['stim', 'time', 'act', 'locx', 'locy']], family=family).fit()
    #model_saturated = sm.GLM(endog=dta['y'],exog=pd.concat([dta[['stim', 'act', 'locx', 'locy']], one_hot_time], axis=1), family=family).fit()
    model_saturated = glm(formula=one_hot_formula, data=data_sat, family=family).fit()
    #model_s = sm.GLM(endog=dta['y'], exog=dta['stim'], family=family).fit()
    #model_t = sm.GLM(endog=dta['y'], exog=dta['time'], family=family).fit()
    #model_a = sm.GLM(endog=dta['y'], exog=dta['act'], family=family).fit()
    #model_p = sm.GLM(endog=dta['y'], exog=dta[['locx', 'locy']], family=family).fit()
    model_s = glm(formula='y ~ stim', data=dta, family=family).fit()
    model_t = glm(formula='y ~ time', data=dta, family=family).fit()
    model_a = glm(formula='y ~ act', data=dta, family=family).fit()
    model_p = glm(formula='y ~ locx + locy', data=dta, family=family).fit()

    # "Modulation" of a certain variable is defined as the LL ratio between the nested model (sans said variable) and the full model

    mod_s = likelihood_ratio(model_tap.llf, model_full.llf)  # deviance of T+A+P model from S+T+A+P model
    mod_t = likelihood_ratio(model_sap.llf, model_full.llf)
    mod_a = likelihood_ratio(model_stp.llf, model_full.llf)
    mod_p = likelihood_ratio(model_sta.llf, model_full.llf)
    dev[i_neuron, :] = [mod_s, mod_t, mod_a, mod_p]
    pr2[i_neuron, :] = [pseudo_r2(model_s, model_saturated),
                        pseudo_r2(model_t, model_saturated),
                        pseudo_r2(model_a, model_saturated),
                        pseudo_r2(model_p, model_saturated),
                        pseudo_r2(model_full, model_saturated)]

    # Log-likelihood ratio test: Akhlaghpour et al., 2016
    p_s = chi2.sf(mod_s, 1)  # L2 has 1 DoF more than L1
    p_t = chi2.sf(mod_t, 1)
    p_a = chi2.sf(mod_a, 1)
    p_p = chi2.sf(mod_p, 1)

    if p_s < mod_sig_threshold:
        s_sig_neurons.append(i_neuron)
    if p_t < mod_sig_threshold:
        t_sig_neurons.append(i_neuron)
    if p_a < mod_sig_threshold:
        a_sig_neurons.append(i_neuron)
    if p_p < mod_sig_threshold:
        p_sig_neurons.append(i_neuron)

dev_labels = ['Sample', 'Time', 'Action', 'Place']

plot_modulation(0, 1, dev_labels, dev)
plot_modulation(0, 2, dev_labels, dev)
plot_modulation(0, 3, dev_labels, dev)
plot_modulation(1, 2, dev_labels, dev)
plot_modulation(1, 3, dev_labels, dev)
plot_modulation(2, 3, dev_labels, dev)

print(f'Number of time-modulated cells: {len(t_sig_neurons)} \n '
      f'Number of location-modulated cells: {len(p_sig_neurons)} \n '
      f'Number of HD-modulated cells: {len(a_sig_neurons)} \n '
      f'Number of sample-modulated cells: {len(s_sig_neurons)}')

# Plot venn
from venn import venn

dict = {
    'Time': set(t_sig_neurons),
    'Location': set(p_sig_neurons),
    'Action': set(a_sig_neurons),
    'Sample': set(s_sig_neurons)
}
fig, ax = plt.subplots()
ax = venn(dict)
plt.title(f'Threshold: {mod_sig_threshold}')
plt.show()

model_labels = ['stim only', 'time only', 'action only', 'place only', 'full']
color_names = ['b', 'g', 'r', 'c', 'm']

# Plotting cumulative density for psuedo-R2 values
fig, ax = plt.subplots(figsize=(10, 10))

for i_model in range(5):
    # if model_labels[i_model] in model_labels_:
    n, bins, patches = ax.hist(pr2[:, i_model], bins=100, density=True, histtype='step',
                               cumulative=True, label=model_labels[i_model], color=color_names[i_model])
ax.legend()
ax.set_xlabel('pseudo-R2')
ax.set_ylabel('cumulative density')
plt.show()
'''
