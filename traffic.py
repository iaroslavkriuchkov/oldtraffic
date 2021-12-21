
from numpy.core.numeric import _moveaxis_dispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pandas.core.algorithms import value_counts
import requests
import datetime
from scipy.stats.mstats import hmean
import os
import pathlib
from pystoned import CQER, wCQER, CQERG
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from pystoned import dataset as dataset
from pystoned.plot import plot2d, plot2d_test
import pyarrow
import sys
import iarotraffic.traffic as iarotr

""" MAIN FUNCTION STARTS HERE"""

def main(output='FILE', do_bagging=True):
    # Select the direction of the road
    select_direction = 2

    # Select the type of output - local file or terminal
    if output == 'FILE':
        original_stdout = sys.stdout
        f = open('log.txt', 'w')
        sys.stdout = f

    # Loading data
    df = iarotr.traffic_data_load('146', '01', 2018, 32, 59)
    test_df = iarotr.traffic_data_load('146', '01', 2019, 32, 59)
    # print(df)

    # Aggregating data
    df_space_agg = iarotr.flow_speed_calculation(df)
    test_df_space_agg = iarotr.flow_speed_calculation(test_df)

    # Bagging data if necessary
    if do_bagging is True:
        grid = iarotr.bagging(df_space_agg)
        test_grid = iarotr.bagging(test_df_space_agg)
        x = grid[grid.direction == select_direction].centroid_density
        y = grid[grid.direction == select_direction].centroid_flow
        w = grid[grid.direction == select_direction].weight
        test_x = test_grid[test_grid.direction == select_direction].centroid_density
        test_y = test_grid[test_grid.direction == select_direction].centroid_flow
        plt.scatter(
            test_grid[test_grid.direction == select_direction].centroid_density,
            test_grid[test_grid.direction == select_direction].centroid_flow,
            c='r',
            marker='o',
            s=test_grid[test_grid.direction == select_direction].weight*10000,
            label=str(select_direction))
        plt.savefig(fname="test_scatter")
    else:
        x = np.array(df_space_agg[df_space_agg.direction == select_direction].density).reshape(
            len(df_space_agg[df_space_agg.direction == select_direction]), 1)
        y = np.array(df_space_agg[df_space_agg.direction == select_direction].flow).reshape(
            len(df_space_agg[df_space_agg.direction == select_direction]), 1)
        test_x = np.array(test_df_space_agg[test_df_space_agg.direction == select_direction].density).reshape(
            len(test_df_space_agg[test_df_space_agg.direction == select_direction]), 1)
        test_y = np.array(test_df_space_agg[test_df_space_agg.direction == select_direction].flow).reshape(
            len(test_df_space_agg[test_df_space_agg.direction == select_direction]), 1)

    test_array = np.column_stack((test_x, test_y))
    print(test_array)

    # plt.show()

    """
    plt.scatter(grid[grid.direction==1].centroid_density, grid[grid.direction==1].centroid_flow,
        c='r', marker = 'o', s=grid[grid.direction==1].weight*10000, label='Direction 1')
    plt.scatter(grid[grid.direction==2].centroid_density, grid[grid.direction==2].centroid_flow,
        c='g', marker = 'o', s=grid[grid.direction==2].weight*10000, label='Direction 2')
    plt.legend()
    plt.show()
    """
    if do_bagging is False:
        model = CQERG.CQRG(y, x, tau=0.5, z=None,
                           cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
        model.optimize(OPT_LOCAL)
        print(model.get_residual())
        print(model.get_runningtime())
        plot2d(model, x_select=0, label_name="Figure 1", fig_name="Figure 1")
    else:
        model = wCQER.wCQR(y=y, x=x, w=w, tau=0.5, z=None,
                           cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
        model.optimize(OPT_LOCAL)

    iarotr.out_of_sample_mse(model, test_array)

    plot2d(model, x_select=0, label_name="Figure 1", fig_name="Figure 1")
    plot2d_test(model, test_array, x_select=0, label_name="Figure 2", fig_name="Figure 2")

    # Close the log-file in case of 'FILE' output
    if output == 'FILE':
        f.close()
        sys.stdout = original_stdout

    pass


if __name__ == "__main__":
    main()
