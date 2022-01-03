
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
from xlwt import Workbook

def processing(train_params: list, test_params: list, output='FILE', do_bagging=True):

    start_time = time.perf_counter()
    file_timestamp = datetime.datetime.now()
    nl = '\n'

    # Write the head for the log
    log_file_name = 'log' + '_' + train_params[0] + '_' + str(train_params[2])[2:4] \
        + '_' + str(train_params[3]) + '_' + str(train_params[4]) + '_' \
        + file_timestamp.strftime("%y%m%d%H%M%S") + '.txt'

    if output == 'FILE':
        original_stdout = sys.stdout
        f = open(log_file_name, 'w')
        sys.stdout = f

    print(f"Created by Iaroslav Kriuchkov")
    print(f"Aalto University School of Business")
    print(f"Department of Information and Service Management\n\n")
    print(f"Log-file for traffic modeling")
    last_update_time = file_timestamp.strftime("%d-%m-%Y %H:%M")
    print(f"Last update: {last_update_time}{nl}{nl}")

    from_date = iarotr.day_to_date(train_params[2], train_params[3]).strftime("%d-%m-%Y")
    to_date = iarotr.day_to_date(train_params[2], train_params[4]).strftime("%d-%m-%Y")
    print(
        f"Train data from TMS #{train_params[0]} "
        f"from {from_date} "
        f"to {to_date}{nl}{nl}")

    from_date = iarotr.day_to_date(test_params[2], test_params[3]).strftime("%d-%m-%Y")
    to_date = iarotr.day_to_date(test_params[2], test_params[4]).strftime("%d-%m-%Y")
    print(
        f"Test data from TMS #{test_params[0]} "
        f"from {from_date} "
        f"to {to_date}{nl}{nl}")

    # Select the direction of the road
    select_direction = 2

    # Loading data
    with open(log_file_name, 'a+'):
        print("Trying to load train data...")
        df = iarotr.traffic_data_load(
            train_params[0], train_params[1], train_params[2], train_params[3], train_params[4])
        print("Trying to load test data...")
        test_df = iarotr.traffic_data_load(
            test_params[0], test_params[1], test_params[2], test_params[3], test_params[4])

    # Aggregating data
    with open(log_file_name, 'a+'):
        print("Aggregating train data...")
        df_space_agg = iarotr.flow_speed_calculation(df)
        print("Aggregating test data...")
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
        scatter_name = 'test_scatter_' + test_params[0] + '_' + str(test_params[2])[2:4] \
            + '_' + str(test_params[3]) + '_' + str(test_params[4]) + '_' \
            + file_timestamp.strftime("%y%m%d%H%M%S")
        plt.savefig(fname=scatter_name)
    else:
        x = np.array(df_space_agg[df_space_agg.direction == select_direction].density).reshape(
            len(df_space_agg[df_space_agg.direction == select_direction]), 1)
        y = np.array(df_space_agg[df_space_agg.direction == select_direction].flow).reshape(
            len(df_space_agg[df_space_agg.direction == select_direction]), 1)
        test_x = np.array(test_df_space_agg[test_df_space_agg.direction == select_direction].density).reshape(
            len(test_df_space_agg[test_df_space_agg.direction == select_direction]), 1)
        test_y = np.array(test_df_space_agg[test_df_space_agg.direction == select_direction].flow).reshape(
            len(test_df_space_agg[test_df_space_agg.direction == select_direction]), 1)
        plt.scatter(
            test_df_space_agg[test_df_space_agg.direction == select_direction].density,
            test_df_space_agg[test_df_space_agg.direction == select_direction].flow,
            c='g',
            marker='x',
            label=str(select_direction))
        scatter_name = 'test_scatter_' + test_params[0] + '_' + str(test_params[2])[2:4] \
            + '_' + str(test_params[3]) + '_' + str(test_params[4]) + '_' \
            + file_timestamp.strftime("%y%m%d%H%M%S")
        plt.savefig(fname=scatter_name)

    test_array = np.column_stack((test_x, test_y))

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
        # plot2d(model, x_select=0, label_name="Figure 1", fig_name="Figure 1")
    else:
        model = wCQER.wCQR(y=y, x=x, w=w, tau=0.5, z=None,
                           cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
        model.optimize(OPT_LOCAL)

    mse = iarotr.out_of_sample_mse(model, test_array)

    fig1 = 'train_data' + '_' + train_params[0] + '_' + str(train_params[2])[2:4] \
        + '_' + str(train_params[3]) + '_' + str(train_params[4]) + '_' \
        + file_timestamp.strftime("%y%m%d%H%M%S")

    fig2 = 'test_data' + '_' + test_params[0] + '_' + str(test_params[2])[2:4] \
        + '_' + str(test_params[3]) + '_' + str(test_params[4]) + '_' \
        + file_timestamp.strftime("%y%m%d%H%M%S")

    plot2d(model, x_select=0, label_name="Figure 1", fig_name=fig1)
    plot2d_test(model, test_array, x_select=0, label_name="Figure 2", fig_name=fig2)

    end_time = time.perf_counter()
    print(f"Execution time: {end_time-start_time:0.4f}")

    if output == 'FILE':
        f.close()
        sys.stdout = original_stdout

    return mse


def processing_diff_models(train_params: list, output='FILE'):

    start_time = time.perf_counter()
    file_timestamp = datetime.datetime.now()
    nl = '\n'

    # Write the head for the log
    log_file_name = 'log' + '_' + train_params[0] + '_' + str(train_params[2])[2:4] \
        + '_' + str(train_params[3]) + '_' + str(train_params[4]) + '_' \
        + file_timestamp.strftime("%y%m%d-%H%M%S") + '.txt'

    if output == 'FILE':
        original_stdout = sys.stdout
        f = open(log_file_name, 'w')
        sys.stdout = f

    print(f"Created by Iaroslav Kriuchkov")
    print(f"Aalto University School of Business")
    print(f"Department of Information and Service Management\n\n")
    print(f"Log-file for traffic modeling")
    last_update_time = file_timestamp.strftime("%d-%m-%Y %H:%M")
    print(f"Last update: {last_update_time}{nl}{nl}")

    from_date = iarotr.day_to_date(train_params[2], train_params[3]).strftime("%d-%m-%Y")
    to_date = iarotr.day_to_date(train_params[2], train_params[4]).strftime("%d-%m-%Y")
    print(
        f"Train data from TMS #{train_params[0]} "
        f"from {from_date} "
        f"to {to_date}.{nl}{nl}")

    print(
        f"Test data is the same as train data.{nl}{nl}")

    # Select the direction of the road
    select_direction = 2

    # Loading data
    with open(log_file_name, 'a+'):
        print("Trying to load train data...")
        df = iarotr.traffic_data_load(
            train_params[0], train_params[1], train_params[2], train_params[3], train_params[4])
        print("Trying to load test data...")
        test_df = df
        print("Test data loaded.")

    # Aggregating data
    with open(log_file_name, 'a+'):
        print("Aggregating train data...")
        df_space_agg = iarotr.flow_speed_calculation(df)
        print("Aggregating test data...")
        test_df_space_agg = df_space_agg

    # Bagging train data
    grid = iarotr.bagging(df_space_agg)

    error_list = iarotr.compare_models(grid, df_space_agg)

    end_time = time.perf_counter()
    print(f"Execution time: {end_time-start_time:0.4f}")

    if output == 'FILE':
        f.close()
        sys.stdout = original_stdout

    return error_list


""" MAIN FUNCTION STARTS HERE"""
def main():
    start_time1 = time.perf_counter()
    wb = Workbook()
    sheet1 = wb.add_sheet('MSE Results')
    train_params = [
        ['146', '01', 2018, 1, 31, 'January'],
        ['146', '01', 2018, 32, 59, 'February'],
        ['146', '01', 2018, 60, 90, 'March'],
        ['146', '01', 2018, 91, 120, 'April'],
        ['146', '01', 2018, 121, 151, 'May'],
        ['146', '01', 2018, 152, 181, 'June'],
        ['146', '01', 2018, 182, 212, 'July'],
        ['146', '01', 2018, 213, 243, 'August'],
        ['146', '01', 2018, 244, 273, 'September'],
        ['146', '01', 2018, 274, 304, 'October'],
        ['146', '01', 2018, 305, 334, 'November'],
        ['146', '01', 2018, 335, 365, 'December']]

    test_params = [
        ['146', '01', 2019, 1, 31, 'January'],
        ['146', '01', 2019, 32, 59, 'February'],
        ['146', '01', 2019, 60, 90, 'March'],
        ['146', '01', 2019, 91, 120, 'April'],
        ['146', '01', 2019, 121, 151, 'May'],
        ['146', '01', 2019, 152, 181, 'June'],
        ['146', '01', 2019, 182, 212, 'July'],
        ['146', '01', 2019, 213, 243, 'August'],
        ['146', '01', 2019, 244, 273, 'September'],
        ['146', '01', 2019, 274, 304, 'October'],
        ['146', '01', 2019, 305, 334, 'November'],
        ['146', '01', 2019, 335, 365, 'December']]

    dir_name = './dir_' + train_params[0][0] + '_' + str(train_params[0][2])[2:4] \
        + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    os.makedirs(dir_name)
    os.chdir(dir_name)
    os.makedirs('./parquetdata')

    sheet1.write(1, 0, 'Train MSE')
    sheet1.write(2, 0, 'Train RMSE')
    sheet1.write(3, 0, 'Train MAE')
    sheet1.write(4, 0, 'Test MSE')
    sheet1.write(5, 0, 'Test RMSE')
    sheet1.write(6, 0, 'Test MAE')

    for count, values in enumerate(train_params):
        start_time = time.perf_counter()
        print(f"Starting the work for {train_params[count]}...")
        mse_list = processing(train_params[count], test_params[count])
        print(mse_list)
        sheet1.write(0, count+1, train_params[count][5])
        sheet1.write(1, count+1, mse_list[0][0])
        sheet1.write(2, count+1, mse_list[0][1])
        sheet1.write(3, count+1, mse_list[0][2])
        sheet1.write(4, count+1, mse_list[1][0])
        sheet1.write(5, count+1, mse_list[1][1])
        sheet1.write(6, count+1, mse_list[1][2])
        end_time = time.perf_counter()
        print(f"Execution time for {train_params[count]}: {end_time-start_time:0.4f}")

    wb.save('MSE.xls')
    end_time1 = time.perf_counter()
    print(f"Execution time: {end_time1-start_time1:0.4f}")
    return None

def main_diff_models():
    start_time1 = time.perf_counter()
    wb = Workbook()
    sheet1 = wb.add_sheet('MSE Results')
    train_params = [
        ['146', '01', 2018, 1, 31, 'January']]

    """
    ,
    ['146', '01', 2018, 32, 59, 'February'],
    ['146', '01', 2018, 60, 90, 'March'],
    ['146', '01', 2018, 91, 120, 'April'],
    ['146', '01', 2018, 121, 151, 'May'],
    ['146', '01', 2018, 152, 181, 'June'],
    ['146', '01', 2018, 182, 212, 'July'],
    ['146', '01', 2018, 213, 243, 'August'],
    ['146', '01', 2018, 244, 273, 'September'],
    ['146', '01', 2018, 274, 304, 'October'],
    ['146', '01', 2018, 305, 334, 'November'],
    ['146', '01', 2018, 335, 365, 'December']]"""

    dir_name = './dir_' + train_params[0][0] + '_' + str(train_params[0][2])[2:4] \
        + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    os.makedirs(dir_name)
    os.chdir(dir_name)
    os.makedirs('./parquetdata')

    sheet1.write(0, 1, 'Bb MSE')
    sheet1.write(0, 2, 'Bb RMSE')
    sheet1.write(0, 3, 'Bb MAE')
    sheet1.write(0, 5, 'Bo MSE')
    sheet1.write(0, 6, 'Bo RMSE')
    sheet1.write(0, 7, 'Bo MAE')
    sheet1.write(0, 9, 'Oo MSE')
    sheet1.write(0, 10, 'Oo RMSE')
    sheet1.write(0, 11, 'Oo MAE')
    sheet1.write(0, 13, 'OB MSE')
    sheet1.write(0, 14, 'OB RMSE')
    sheet1.write(0, 15, 'OB MAE')

    for count, values in enumerate(train_params):
        start_time = time.perf_counter()
        print(f"Starting the work for {train_params[count]}...")
        mse_list = processing_diff_models(train_params[count])
        print(mse_list)
        sheet1.write(count+1, 0, train_params[count][5])
        sheet1.write(count+1, 1, mse_list[0][0])
        sheet1.write(count+1, 2, mse_list[0][1])
        sheet1.write(count+1, 3, mse_list[0][2])
        sheet1.write(count+1, 5, mse_list[1][0])
        sheet1.write(count+1, 6, mse_list[1][1])
        sheet1.write(count+1, 7, mse_list[1][2])
        sheet1.write(count+1, 9, mse_list[2][0])
        sheet1.write(count+1, 10, mse_list[2][1])
        sheet1.write(count+1, 11, mse_list[2][2])
        sheet1.write(count+1, 13, mse_list[3][0])
        sheet1.write(count+1, 14, mse_list[3][1])
        sheet1.write(count+1, 15, mse_list[3][2])
        end_time = time.perf_counter()
        print(f"Execution time for {train_params[count]}: {end_time-start_time:0.4f}")

    wb.save('MSE.xls')
    end_time1 = time.perf_counter()
    print(f"Execution time: {end_time1-start_time1:0.4f}")
    return None

# if __name__ == "__main__":
#    main()


main_diff_models()
