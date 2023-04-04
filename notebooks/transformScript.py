import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_market_calendars as mcal
import os


def read_file_and_export(f: str, export: str) -> None:
    """
    Read the file and export the data to a csv file so that it can be read by pandas
    and transformed to the proper format
    :param f:
    :param export:
    :return:
    """

    count = 0
    import csv
    file = open(f, 'r')
    file2 = open(export, 'x')
    writer = csv.writer(file2)
    columns = ['Date__(UT)__HR:MN', 'R.A._(ICRF)', 'DEC__(ICRF)', 'APmag',
               'S-brt', 'Illu%', 'hEcl-Lon', 'hEcl-Lat', 'r', 'rdot', 'delta', 'deldot',
               'S-O-T', '/r', 'S-T-O', 'GlxLon', 'GlxLat', 'L_Ap_SOL_Time', 'RA_3sigma', 'DEC_3sigma',
               'SMAA_3sig', 'SMIA_3sig', 'Theta', 'Area_3sig', 'POS_3sigma', 'RNG_3sigma',
               'RNGRT_3sig', 'DOP_S_3sig', 'DOP_X_3sig', 'RT_delay_3sig', 'App_Lon_Sun',
               'RA_(ICRF-a-app)', 'DEC_(ICRF-a-app)', 'Sky_motion', 'Sky_mot_PA', 'RelVel-ANG',
               'Lun_Sky_Brt', 'sky_SNR']
    writer.writerow(columns)
    while True:
        line = file.readline()

        if not line:
            break
        else:
            # print('yes')
            if '$$SOE' in line:

                count += 1
            elif '$$EOE' in line:
                count += 1
            if count == 1 and '$$SOE' not in line:
                line = line.replace(' ', '')
                line = line.split(',')
                line.remove('')
                line.remove('')

                writer.writerow(line)
    file.close()
    file2.close()


def filter_by_time(df, start, end) -> pd.DataFrame:
    """
    Filter the data by time
    :param df: the dataframe
    :param start: start time
    :param end: end time
    :return: the filtered dataframe
    """

    df.query('time >= @start and time <= @end', inplace=True)

    return df


import datetime


def filter_by_date(df, start, end) -> pd.DataFrame:
    """
    Filter the data by date
    :param df: the dataframe
    :param start: start date (string in format 'yyyy-mm-dd')
    :param end: end date (string in format 'yyyy-mm-dd')
    :return: the filtered dataframe
    """
    # Convert start and end to datetime.date objects
    start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()

    # Filter the dataframe
    df.query('date >= @start_date and date <= @end_date', inplace=True)

    return df


def read_csv_transform(f: str, export: str) -> pd.DataFrame:
    """
    Read the csv file and transform it to the proper format for the model
    :param f: file path
    :param export: export path
    :return: the transformed dataframe
    """
    data = pd.read_csv(f, na_values='n.a.', low_memory=False)
    data.dropna(axis=1, inplace=True)
    data.drop('RA_(ICRF-a-app)', axis=1, inplace=True)
    data.drop('R.A._(ICRF)', axis=1, inplace=True)

    data['/r'] = data['/r'].replace({'/T': 1, '/L': 0})

    data['Date__(UT)__HR:MN'] = pd.to_datetime(data['Date__(UT)__HR:MN'], format='%Y-%b-%d%H:%M')
    data['year'] = data['Date__(UT)__HR:MN'].dt.year
    data['month'] = data['Date__(UT)__HR:MN'].dt.month
    data['day'] = data['Date__(UT)__HR:MN'].dt.day
    data['time'] = data['Date__(UT)__HR:MN'].dt.hour + data['Date__(UT)__HR:MN'].dt.minute / 60
    data['date'] = data['Date__(UT)__HR:MN'].dt.date
    # data['date'] = pd.to_datetime(data['date'])
    filter_by_date(data, '2020-06-27', '2023-03-10')
    data['weekday'] = data['Date__(UT)__HR:MN'].dt.dayofweek
    data = data[data['weekday'] < 5]

    work_hour_df = data[(data['time'] >= 9.5) & (data['time'] <= 15.5)]
    # filter so data goes by hour from 9:30 to 14:30
    work_hour_df = work_hour_df[work_hour_df['time'] % 1 == 0.5]


    #cal = USFederalHolidayCalendar()
    cal = mcal.get_calendar('NYSE')
    #data['Date__(UT)__HR:MN'] = pd.to_datetime(data['Date__(UT)__HR:MN'], format='%d,%m,%Y %H:%M')

    holidays = cal.schedule(start_date=work_hour_df['Date__(UT)__HR:MN'].min(), end_date=work_hour_df['Date__(UT)__HR:MN'].max())
    holidays = pd.to_datetime(holidays.index, format='%Y-%b-%d%H:%M')

    work_hour_df = work_hour_df[pd.to_datetime(work_hour_df['date']).isin(holidays)]

    # work_hour_df.drop('Date__(UT)__HR:MN', axis = 1, inplace=True)

    return work_hour_df


def main():
    read_file_and_export('../Data/planets.txt', 'data.csv')
    df = read_csv_transform('data.csv', 'data.csv')
    df.to_csv('data.csv', index=False, )


if __name__ == '__main__':
    main()
