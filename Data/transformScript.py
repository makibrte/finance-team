import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar


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


def read_csv_transform(f: str, export: str) -> pd.DataFrame:
    """
    Read the csv file and transform it to the proper format for the model
    :param f: file path
    :param export: export path
    :return: the transformed dataframe
    """
    data = pd.read_csv(f, na_values='n.a.')
    data.dropna(axis=1, inplace=True)
    data.drop('Lun_Sky_Brt', axis=1, inplace=True)
    data.drop('R.A._(ICRF)', axis=1, inplace=True)

    data['/r'] = data['/r'].replace({'/T': 1, '/L': 0})

    data['Date__(UT)__HR:MN'] = pd.to_datetime(data['Date__(UT)__HR:MN'], format='%Y-%b-%d%H:%M')
    data['year'] = data['Date__(UT)__HR:MN'].dt.year
    data['month'] = data['Date__(UT)__HR:MN'].dt.month
    data['day'] = data['Date__(UT)__HR:MN'].dt.day
    data['time'] = data['Date__(UT)__HR:MN'].dt.hour + data['Date__(UT)__HR:MN'].dt.minute / 60
    data['date'] = data['Date__(UT)__HR:MN'].dt.date
    data['date'] = pd.to_datetime(data['date'])

    data['weekday'] = data['Date__(UT)__HR:MN'].dt.dayofweek
    data = data[data['weekday'] < 5]

    work_hour_df = filter_by_time(data, 9, 17)

    cal = USFederalHolidayCalendar()

    holidays = cal.holidays(start=work_hour_df['Date__(UT)__HR:MN'].min(), end=work_hour_df['Date__(UT)__HR:MN'].max())
    holidays = pd.to_datetime(holidays, format='%Y-%b-%d%H:%M')

    work_hour_df = work_hour_df[~work_hour_df['date'].isin(holidays)]

    return work_hour_df

def main():
    read_file_and_export('../Data/planets.txt', 'data.csv')
    df = read_csv_transform('data.csv', 'data.csv')
    df.to_csv('data.csv', index=False)

if __name__ == '__main__':
    main()

