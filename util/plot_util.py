import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.pyplot import cm

def plot_grid_results(results, x_var, legend_var=None, log=False):
    """
    Plot the results of a grid search as a function of one of the parameters.
    
    Parameters:
    - results (GridSearchCV): The results of the grid search.
    - x_var (str): The name of the parameter to plot on the x-axis.
    - legend_var (str, optional): The name of the parameter to use in the legend. If not provided, no legend will be shown.
    - log (bool, optional): Whether to plot the x-axis on a log scale.
    
    Returns:
    - None
    """
    if legend_var is None:
        legend_values = ['dummy']
    else:
        legend_values = results.param_grid[legend_var]
    for l_val in legend_values:
        scores = results.cv_results_['mean_test_score']
        x_values = results.cv_results_[f'param_{x_var}'].data
        if legend_var:
            mask = [item[legend_var] == l_val for item in results.cv_results_['params']]
            scores = scores[mask]
            x_values = x_values[mask]
        if log:
            axes = plt.subplot()
            axes.set_xscale('log')
        plt.plot(x_values, scores, label=f'{legend_var} = {l_val}' if legend_var else '')
    plt.xlabel(x_var)
    plt.ylabel(results.scoring)
    if legend_var is not None:
        plt.legend()
