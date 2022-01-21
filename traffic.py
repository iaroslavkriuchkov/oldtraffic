
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

    error_list = iarotr.compare_models(grid, df_space_agg, train_params[5], train_params[2], train_params[6])

    end_time = time.perf_counter()
    print(f"Execution time: {end_time-start_time:0.4f}")

    if output == 'FILE':
        f.close()
        sys.stdout = original_stdout

    return error_list


""" MAIN FUNCTION STARTS HERE"""
def main():
    start_time1 = time.perf_counter()
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
        ['146', '01', 2018, 335, 365, 'December'],
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

    for count, values in enumerate(train_params):

        df = iarotr.traffic_data_load(
            train_params[count][0], train_params[count][1],
            train_params[count][2], train_params[count][3], train_params[count][4])
        df = iarotr.flow_speed_calculation(df)

        fig_name = "./graphs/Original data scatter-" + str(count)
        plt.scatter(
            df[df.direction == 2].density,
            df[df.direction == 2].flow,
            c=df[df.direction == 2].car_proportion,
            marker='.',
            cmap="RdYlGn",
            label="Original data scatter")
        plt.savefig(fname=fig_name)
        plt.clf()

    return None

def main_diff_models():
    start_time1 = time.perf_counter()
    wb = Workbook()
    sheet1 = wb.add_sheet('MSE Results')
    tau_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    train_params = [
        ['146', '01', 2018, 1, 31, 'January', 0.5],
        ['146', '01', 2018, 32, 59, 'February', 0.5],
        ['146', '01', 2018, 60, 90, 'March', 0.5],
        ['146', '01', 2018, 91, 120, 'April', 0.5],
        ['146', '01', 2018, 121, 151, 'May', 0.5],
        ['146', '01', 2018, 152, 181, 'June', 0.5]]

    """['146', '01', 2018, 182, 212, 'July'],
        ['146', '01', 2018, 213, 243, 'August'],
        ['146', '01', 2018, 244, 273, 'September'],
        ['146', '01', 2018, 274, 304, 'October'],
        ['146', '01', 2018, 305, 334, 'November'],
        ['146', '01', 2018, 335, 365, 'December'],
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
        ['146', '01', 2019, 335, 365, 'December']"""

    dir_name = './dir_' + train_params[0][0] + '_' + str(train_params[0][2])[2:4] \
        + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    os.makedirs(dir_name)
    os.chdir(dir_name)
    sheet1.write(0, 0, 'Month')
    sheet1.write(0, 1, 'tau')
    sheet1.write(0, 3, 'Bb MSE')
    sheet1.write(0, 4, 'Bb RMSE')
    sheet1.write(0, 5, 'Bb MAE')
    sheet1.write(0, 7, 'Bo MSE')
    sheet1.write(0, 8, 'Bo RMSE')
    sheet1.write(0, 9, 'Bo MAE')
    sheet1.write(0, 11, 'Oo MSE')
    sheet1.write(0, 12, 'Oo RMSE')
    sheet1.write(0, 13, 'Oo MAE')
    sheet1.write(0, 15, 'OB MSE')
    sheet1.write(0, 16, 'OB RMSE')
    sheet1.write(0, 17, 'OB MAE')
    sheet1.write(0, 19, 'max Density bagged')
    sheet1.write(0, 20, 'max Flow bagged')
    sheet1.write(0, 22, 'max Density orig')
    sheet1.write(0, 23, 'max Flow orig')

    row = 1

    for count, values in enumerate(train_params):
        month_dir_name = './' + train_params[count][5] + "_" + str(train_params[count][2])
        os.makedirs(month_dir_name)
        os.chdir(month_dir_name)
        for tau in tau_list:
            start_time = time.perf_counter()
            tau_dir_name = './tau_' + str(int(tau*100))
            os.makedirs(tau_dir_name)
            os.chdir(tau_dir_name)
            train_params[count][6] = tau
            mse_list = processing_diff_models(train_params[count])
            cell1 = train_params[count][5] + " " + str(train_params[count][2])
            cell2 = str(train_params[count][6])
            sheet1.write(row, 0, cell1)
            sheet1.write(row, 1, cell2)
            sheet1.write(row, 3, mse_list[0][0])
            sheet1.write(row, 4, mse_list[0][1])
            sheet1.write(row, 5, mse_list[0][2])
            sheet1.write(row, 7, mse_list[1][0])
            sheet1.write(row, 8, mse_list[1][1])
            sheet1.write(row, 9, mse_list[1][2])
            sheet1.write(row, 11, mse_list[2][0])
            sheet1.write(row, 12, mse_list[2][1])
            sheet1.write(row, 13, mse_list[2][2])
            sheet1.write(row, 15, mse_list[3][0])
            sheet1.write(row, 16, mse_list[3][1])
            sheet1.write(row, 17, mse_list[3][2])
            sheet1.write(row, 19, mse_list[4][0])
            sheet1.write(row, 20, mse_list[4][1])
            sheet1.write(row, 22, mse_list[5][0])
            sheet1.write(row, 23, mse_list[5][1])
            os.chdir('../')
            row += 1
            end_time = time.perf_counter()
        os.chdir('../')

    wb.save('MSE.xls')
    end_time1 = time.perf_counter()
    print(f"Execution time: {end_time1-start_time1:0.4f}")
    return None


def days_prediction(lam_id: str, region: str, pred_year: int, pred_day: int, num_days: int) -> None:
    start_time1 = time.perf_counter()
    wb = Workbook()
    sheet1 = wb.add_sheet('MSE Results')
    tau_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    days_list = iarotr.previous_days(pred_year, pred_day, num_days)
    train_params = [None] * num_days
    for i in range(num_days):
        train_params[i] = [
            lam_id,
            region,
            days_list[i][0],
            days_list[i][1],
            days_list[i][1],
            iarotr.day_to_date(days_list[i][0], days_list[i][1]).isoformat(), 0.5]

    dir_name = './dir_' + train_params[0][0] + '_' + str(train_params[0][2])[2:4] \
        + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    os.makedirs(dir_name)
    os.chdir(dir_name)
    sheet1.write(0, 0, 'Month')
    sheet1.write(0, 1, 'tau')
    sheet1.write(0, 3, 'Bb MSE')
    sheet1.write(0, 4, 'Bb RMSE')
    sheet1.write(0, 5, 'Bb MAE')
    sheet1.write(0, 7, 'Bo MSE')
    sheet1.write(0, 8, 'Bo RMSE')
    sheet1.write(0, 9, 'Bo MAE')
    sheet1.write(0, 11, 'Oo MSE')
    sheet1.write(0, 12, 'Oo RMSE')
    sheet1.write(0, 13, 'Oo MAE')
    sheet1.write(0, 15, 'OB MSE')
    sheet1.write(0, 16, 'OB RMSE')
    sheet1.write(0, 17, 'OB MAE')
    sheet1.write(0, 19, 'max Density bagged')
    sheet1.write(0, 20, 'max Flow bagged')
    sheet1.write(0, 22, 'max Density orig')
    sheet1.write(0, 23, 'max Flow orig')

    row = 1

    for count, values in enumerate(train_params):
        month_dir_name = './' + train_params[count][5] + "_" + str(train_params[count][2])
        os.makedirs(month_dir_name)
        os.chdir(month_dir_name)
        for tau in tau_list:
            start_time = time.perf_counter()
            tau_dir_name = './tau_' + str(int(tau*100))
            os.makedirs(tau_dir_name)
            os.chdir(tau_dir_name)
            train_params[count][6] = tau
            mse_list = processing_diff_models(train_params[count])
            cell1 = train_params[count][5] + " " + str(train_params[count][2])
            cell2 = str(train_params[count][6])
            sheet1.write(row, 0, cell1)
            sheet1.write(row, 1, cell2)
            sheet1.write(row, 3, mse_list[0][0])
            sheet1.write(row, 4, mse_list[0][1])
            sheet1.write(row, 5, mse_list[0][2])
            sheet1.write(row, 7, mse_list[1][0])
            sheet1.write(row, 8, mse_list[1][1])
            sheet1.write(row, 9, mse_list[1][2])
            sheet1.write(row, 11, mse_list[2][0])
            sheet1.write(row, 12, mse_list[2][1])
            sheet1.write(row, 13, mse_list[2][2])
            sheet1.write(row, 15, mse_list[3][0])
            sheet1.write(row, 16, mse_list[3][1])
            sheet1.write(row, 17, mse_list[3][2])
            sheet1.write(row, 19, mse_list[4][0])
            sheet1.write(row, 20, mse_list[4][1])
            sheet1.write(row, 22, mse_list[5][0])
            sheet1.write(row, 23, mse_list[5][1])
            os.chdir('../')
            row += 1
            end_time = time.perf_counter()
        os.chdir('../')

    wb.save('MSE.xls')
    end_time1 = time.perf_counter()
    print(f"Execution time: {end_time1-start_time1:0.4f}")
    return None

# if __name__ == "__main__":
#    main()


main_diff_models()
