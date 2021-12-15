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
from pystoned.plot import plot2d
import pyarrow
import sys

default_column_names = ['id', 'year', 'day', 'hour',
        'minute', 'second', 'hund_second', 'length', 'lane',
        'direction', 'vehicle', 'speed', 'faulty', 'total_time',
        'time_interval', 'queue_start']
default_filepath = 'parquetdata'
default_aggregation_time_period = 5


def download_lam_day_report (tms_id, region, year, day, time_from=6, time_to=20):
    start_time = time.perf_counter() 
    column_names = default_column_names
    df = pd.DataFrame()
    url = 'https://aineistot.liikennevirasto.fi/lam/rawdata/YYYY/REGION_ID/lamraw_TMS_YY_DD.csv'

    # Create the actual url
    url = url.replace('YYYY', str(year)).replace('REGION_ID', region).replace('TMS', tms_id).replace('YY', str(year)[2:4]).replace('DD', str(day))

    # Try to download the file
    if requests.get(url).status_code != 404:
        
        # Downloading the file from the server
        df = pd.read_csv(url, delimiter = ";", names = column_names)
        
        # Assigning dates
        df['date'] = datetime.date(year, 1, 1) + datetime.timedelta(day - 1)

        # Deleting faulty data point.
        df = df[df.faulty != 1]

        # Selecting data only from the specified timeframe
        df = df[df.total_time >= time_from*60*60*100]
        df = df[df.total_time <= time_to*60*60*100]

        end_time = time.perf_counter()
        print(f"Download successful - file for the sensor {tms_id} for the day {day} in year {year}. \
                Download took {end_time-start_time:0.4f} seconds")
    else:
        print(f"File for the sensor {tms_id} for the day {day} in year {year} doesn't exist. ")
        return df

    return df

"""LOADING TRAFFIC DATA FROM LOCAL GZIP-FILE. IF UNSUCCESSFUL, LOADS DATA FROM SERVER AND SAVES LOCALLY AS GZIP-FILE"""
def traffic_data_load(tms_id, region, year, day_from, day_to, time_from=6, time_to=20, input_path= None, input_name= None, file_type='gzip'):
    start_time = time.perf_counter() 
    filename = 'data' + '_' + tms_id + '_' + str(year)[2:4] \
            + '_' + str(day_from) + '_' + str(day_to) + '_' + str(time_from) \
            + 'h_' + str(time_to) + 'h.gzip'
    filepath = default_filepath
    column_names = default_column_names
    df = pd.DataFrame()

    # Alternative name selection
    if input_name is not None:
        filename = input_name

    # Alternative name selection
    if input_path is not None:
        filepath = input_path
    
    # Make a unified location identifier
    filepath = pathlib.Path(filepath)
    path = filepath / filename

    # Use the gzip file
    if file_type == 'gzip':

        # First trying to load file locally
        if (os.path.exists(path) is True) and (os.path.getsize(path) != 0):
            df = pd.read_parquet(path)
        else:
            print(f"File {path} doesn't exist. Trying to download data from the server and save locally...")

            # Second trying to download files from online
            start_time_gzip = time.perf_counter() 
            for day in range(day_from, day_to + 1):
                if df.empty:
                    df = download_lam_day_report(tms_id, region, year, day, time_from=time_from, time_to=time_to)
                else:
                    df = df.append(download_lam_day_report(tms_id, region, year, day, time_from=time_from, time_to=time_to), ignore_index=True)
            end_time_gzip = time.perf_counter() 
            print(f"Loading file from server took {end_time_gzip-start_time_gzip:0.4f} seconds. Saving .gzip file...")
            
            # Saving the .gzip file with downloaded data
            start_time_gzip = time.perf_counter() 
            df.to_parquet(path, engine='pyarrow', compression='gzip')
            end_time_gzip = time.perf_counter() 
            print(f"Saving .gzip file took {end_time_gzip-start_time_gzip:0.4f} seconds")

    
    end_time = time.perf_counter()
    if df.empty:
        print(f"Loading unsuccessfull. Check the parameters and avaiability of data.")
    else:
        print(f"Loading successfull. Data loading took {end_time-start_time:0.4f} seconds")
    
    # The Pandas DataFrame is returned, containing the data for the selected period
    return df

""" PROCESSING OF THE TRAFFIC DATA FRAME: CALCULATION OF SPACE-MEAN SPEED AND SPACE-MEAN FLOW. 
    BASED ON THAT THE DENSITY IS CALCULATED """

def flow_speed_calculation (df, aggregation_time_period = default_aggregation_time_period):
    start_time = time.perf_counter()
    time_agg = pd.DataFrame()
    space_agg = pd.DataFrame()

    # Create the aggregation parametere based on aggregation_time_period
    df['aggregation'] = (df.hour * 60 + df.minute)/aggregation_time_period
    df = df.astype({'aggregation':int})  

    # Aggregate flow and speed by time
    time_agg = df.groupby(['id','date', 'aggregation', 'direction', 'lane'],
        as_index = False).agg(time_mean_speed=('speed','mean'),
        flow = ('speed','count'))
    time_agg['hourlyflow'] = 60/aggregation_time_period * time_agg.flow
    time_agg['qdivv'] = time_agg['hourlyflow'].div(time_agg['time_mean_speed'].values)

    # Aggregate flow and speed by space and calculate density. Calculate the weights
    space_agg = time_agg.groupby(['id','date', 'aggregation', 'direction'],
        as_index = False).agg(qdivvsum = ('qdivv', 'sum'),
        flow = ('hourlyflow', 'sum'))
    space_agg['space_mean_speed'] = 1/(space_agg.qdivvsum.div(space_agg.flow))
    space_agg['density'] = space_agg.flow.div(space_agg.space_mean_speed)
    space_agg['weight'] = float(1/len(space_agg))

    end_time = time.perf_counter()
    print(f"Aggregating data for modeling took {end_time-start_time:0.4f} seconds")
    
    return space_agg

""" BAGGING OF PROCESSED DATA """
def bagging(aggregative, grid_size_x=40, grid_size_y=40):
    grid = pd.DataFrame()
    
    # Getting the max density and flow values to calculcate the size of the bag
    maxDensity = aggregative.density.max()
    maxFlow = aggregative.flow.max()
    
    # Calclulating the size of the bag
    grid_density_size = maxDensity / grid_size_x
    grid_flow_size = maxFlow / grid_size_y

    # Assigning the bag number for density and
    aggregative['grid_density'] = aggregative.density / grid_density_size
    aggregative['grid_flow'] = aggregative.flow / grid_flow_size
    aggregative = aggregative.astype({'grid_density':int, 'grid_flow':int})

    # Calculating the centroid adn the weight of each bag
    grid = aggregative.groupby(['id','direction', 'grid_density', 'grid_flow'],
        as_index = False).agg(bag_size = ('id', 'count'), 
            sum_flow = ('flow', 'sum'), 
            sum_density = ('density', 'sum'))
    grid['centroid_flow'] = grid.sum_flow.div(grid.bag_size)
    grid['centroid_density'] = grid.sum_density.div(grid.bag_size)
    grid['weight'] = grid.bag_size.div(len(aggregative))

    # Separating the directions
    #grid_dir_1 = grid[grid.direction == 1]
    #grid_dir_2 = grid[grid.direction == 2]
   
    return grid #grid_dir_1, grid_dir_2

""" MAIN FUNCTION STARTS HERE"""
def main(output = 'TERMINAL', do_bagging = True):
    # Select the direction of the road
    select_direction = 2

    # Select the type of output - local file or terminal 
    if output == 'FILE':
        original_stdout = sys.stdout
        f = open('log.txt', 'w')
        sys.stdout = f

    # Loading data
    df = traffic_data_load('146', '01', 2019, 42, 57)
    print(df)

    # Aggregating data
    df_space_agg = flow_speed_calculation(df)

    # Bagging data if necessary
    if do_bagging is True:
        grid = bagging(df_space_agg)
        x = grid[grid.direction==select_direction].centroid_density
        y = grid[grid.direction==select_direction].centroid_flow
        w = grid[grid.direction==select_direction].weight
    else:
        x = np.array(df_space_agg[df_space_agg.direction==select_direction].density).reshape(len(df_space_agg[df_space_agg.direction==select_direction]), 1)
        y = np.array(df_space_agg[df_space_agg.direction==select_direction].flow).reshape(len(df_space_agg[df_space_agg.direction==select_direction]), 1)

    #plt.show()
    
    
    plt.scatter(grid[grid.direction==1].centroid_density, grid[grid.direction==1].centroid_flow,
        c='r', marker = 'o', s=grid[grid.direction==1].weight*10000, label='Direction 1')
    plt.scatter(grid[grid.direction==2].centroid_density, grid[grid.direction==2].centroid_flow,
        c='g', marker = 'o', s=grid[grid.direction==2].weight*10000, label='Direction 2')
    plt.legend()
    plt.show()
    """
    if do_bagging is False:
        model = CQERG.CQRG(y, x, tau=0.5, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
        model.optimize(OPT_LOCAL)
        print(model.get_residual())
        print(model.get_runningtime())
        plot2d(model, x_select=0, label_name="Figure 1", fig_name="Figure 1")
    else:
        model = wCQER.wCQR(y=y, x=x, w=w, tau=0.5, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
        model.optimize(OPT_LOCAL)
        print(model.get_residual())
        plot2d(model, x_select=0, label_name="Figure 1", fig_name="Figure 1")
    """
    # Close the log-file in case of 'FILE' output 
    if output == 'FILE':
        f.close()
        sys.stdout = original_stdout

    pass


if __name__ == "__main__":
    main()

