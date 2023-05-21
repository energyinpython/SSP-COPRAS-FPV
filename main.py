import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from ssp_copras import COPRAS

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights


def main():

    # Load decision matrix with performance values
    df = pd.read_csv('dataset/data.csv', index_col='Technology')
    types = np.array([-1, 1, 1, 1, -1, 1, 1, -1, -1])
    matrix = df.to_numpy()
    old_matrix = copy.deepcopy(matrix)

    # procedure performed because some values in column with index 5 are negative
    matrix[:,5] = matrix[:,5] + np.abs(np.min(matrix[:,5]))

    names = list(df.index)

    # analysis with criteria weights modification

    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)

    # 5 technical criteria, 4 economic criteria

    w_tech_total = np.arange(0.25, 0.8, 0.05)
    w_econ_total = []

    weights_tab = np.zeros((1, 9))

    for wt in w_tech_total:
    
        we = 1 - wt
        w_econ_total.append(we)

        weights = np.zeros(9)
        # technical 5 [0, 1, 2, 3, 4]
        weights[:5] = wt / 5
        # economical 4 [5, 6, 7, 8]
        weights[5:] = we / 4

        weights_tab = np.concatenate((weights_tab, weights.reshape(1, -1)), axis = 0)

        # sustainability coefficient from matrix calculated based on standard deviation from normalized matrix
        n_matrix = norms.minmax_normalization(old_matrix, types)
        s = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])


        ssp_copras = COPRAS(normalization_method=norms.sum_normalization)
        pref = ssp_copras(matrix, weights, types, s_coeff=s)
        rank = rank_preferences(pref, reverse = True)

        results_pref[str(wt)] = pref
        results_rank[str(wt)] = rank


    results_pref = results_pref.rename_axis('Technology')
    results_rank = results_rank.rename_axis('Technology')
    results_pref.to_csv('./results/df_pref_weights.csv')
    results_rank.to_csv('./results/df_rank_weights.csv')

    weights_tab_df = pd.DataFrame(weights_tab)
    weights_tab_df = weights_tab_df.iloc[1:,:]
    weights_tab_df['Technical'] = w_tech_total
    weights_tab_df['Economical'] = w_econ_total
    weights_tab_df.to_csv('./results/weights.csv')


    # plot results of analysis with criteria weights modification
    ticks = np.arange(1, 6)

    x1 = np.arange(0, len(w_tech_total))

    plt.figure(figsize = (10, 6))
    for i in range(results_rank.shape[0]):
        plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                        fontsize = 12, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Technical criteria importance rate", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.xticks(x1, np.round(w_tech_total, 2), fontsize = 12)
    plt.yticks(ticks, fontsize = 12)
    plt.xlim(x_min - 0.2, x_max + 2.4)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('Floating photovoltaic systems rankings')
    plt.tight_layout()
    plt.savefig('./results/technology_rankings_weights.png')
    plt.savefig('./results/technology_rankings_weights.pdf')
    plt.show()

    # analysis with sustainability coefficient modification
    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)


    sust_coeff = np.arange(0, 1.1, 0.1)

    for sc in sust_coeff:

        weights = mcda_weights.equal_weighting(matrix)

        s = np.ones(9) * sc

        ssp_copras = COPRAS(normalization_method=norms.sum_normalization)
        pref = ssp_copras(matrix, weights, types, s_coeff=s)
        rank = rank_preferences(pref, reverse = True)

        results_pref[str(sc)] = pref
        results_rank[str(sc)] = rank


    results_pref = results_pref.rename_axis('Technology')
    results_rank = results_rank.rename_axis('Technology')
    results_pref.to_csv('./results/df_pref_sust.csv')
    results_rank.to_csv('./results/df_rank_sust.csv')

    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(1, 6)

    x1 = np.arange(0, len(sust_coeff))

    plt.figure(figsize = (10, 6))
    for i in range(results_rank.shape[0]):
        plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                        fontsize = 12, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Sustainability coeffcient", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.xticks(x1, np.round(sust_coeff, 2), fontsize = 12)
    plt.yticks(ticks, fontsize = 12)
    plt.xlim(x_min - 0.2, x_max + 2.4)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('Floating photovoltaic systems rankings')
    plt.tight_layout()
    plt.savefig('./results/technology_rankings_sust.png')
    plt.savefig('./results/technology_rankings_sust.pdf')
    plt.show()



if __name__ == '__main__':
    main()