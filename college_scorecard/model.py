'''
features = ['SAT_AVG_ALL']

Test loss: 49939391.2836
Features: ['SAT_AVG_ALL']
Number of examples: 9826

features = ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID', 'SATWRMID']

Test loss: 68742027.3115
Features: ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID', 'SATWRMID']
Number of examples: 3253

'''

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.regularizers import l2
# from keras.utils import np_utils
# import matplotlib.pyplot as plt
# import seaborn as sns
import keras
import pandas
import numpy as np
import glob
import os


def get_clean_data(column):
    # Replace bad values with -1
    column.replace('PrivacySuppressed', -1, inplace=True)
    column.fillna(-1, inplace=True)
    return column.values.astype(np.float)


def print_overall_average(file, data, name):
    aver = data[data != -1]
    overall_aver = np.average(aver) if aver.size > 0 else 'No data'
    print("Year: %s, Average %s: %s, Data points: %d" %
          (file.split('/')[-1][6:-7], name, overall_aver, aver.size))


def get_data(features):
    data_filtered = {}
    data = {}

    cwd = os.getcwd()
    files = sorted(glob.glob(cwd + '/CollegeScorecard_Raw_Data/*.csv'))
    # files = [cwd + '/CollegeScorecard_Raw_Data/MERGED2013_14_PP.csv']
    for file in files:
        # Consider only data from 2xxx (previous years don't have the data we are interested in)
        if file.split('/')[-1][6:-7].startswith('1'):
            continue
        df = pandas.read_csv(file, delimiter=',', error_bad_lines=False)
        # Extract year from file name
        year = file.split('/')[-1][6:-7]
        # Average earnings by male after 6 years of entry
        male_earnings = get_clean_data(df['MN_EARN_WNE_MALE1_P6'])
        # Male count
        male_count = get_clean_data(df['COUNT_WNE_MALE1_P6'])

        # Average earnings by male after 6 years of entry
        female_earnings = get_clean_data(df['MN_EARN_WNE_MALE0_P6'])
        # Female count
        female_count = get_clean_data(df['COUNT_WNE_MALE0_P6'])

        # Average earnings (female and male)
        average_earnings = []
        for (me, mc, fe, fc) in zip(male_earnings, male_count, female_earnings, female_count):
            average_earnings.append((me * mc + fe * fc) / (mc + fc) if -1 not in (me, mc, fe, fc) else -1.0)
        # Convert list to numpy array
        average_earnings = np.asarray(average_earnings)
        data['Y'] = average_earnings
        # Print overall earning average
        print_overall_average(file, average_earnings, 'earnings')

        for feature in features:
            # Get feature data
            feature_data = get_clean_data(df[feature])
            data[feature] = feature_data
            # Print overall feature average
            print_overall_average(file, feature_data, feature)

        # Convert to dataframe
        dataset_df = pandas.DataFrame(data)
        # Filter missing values
        for col in dataset_df.columns:
            query = '%s>-1.0' % col
            dataset_df= dataset_df.query(query)
        dataset_filtered = dataset_df.values
        # Skip years with no data
        if dataset_filtered.size > 0:
            data_filtered[year] = dataset_filtered
    return data_filtered


def train_model(features):
    n_features = len(features)

    dataset = get_data(features)
    data = None
    for key, value in dataset.items():
        if data is None:
            data = value
        else:
            data = np.concatenate((data, value), axis=0)

    # data = dataset['2013_14']
    X, Y = data[:, :n_features], data[:, -1:]

    # Make test and train set
    train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)

    # Linear regression
    model = Sequential()
    model.add(Dense(1, input_shape=(n_features,), activation="linear"))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.1))
    print(model.summary())

    # train
    history = model.fit(train_X, train_y,
                        batch_size=16,
                        epochs=300,
                        verbose=1,
                        validation_data=(test_X, test_y))

    # Evaluate and print MSE
    score = model.evaluate(test_X, test_y, verbose=0)
    print('Test loss: %.4f' % score)
    print("Features: %s" % features)
    print("Number of examples: %d" % len(Y))

    # # Regression plot
    # ax = sns.regplot(x='X', y='Y', data=pandas.DataFrame({'X': test_X, 'Y': test_y}))
    # ax.set(xlabel='Average SAT', ylabel='Mean earnings 6 years after entry')
    # plt.show()

    return score

if __name__ == '__main__':
    features = ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID'] #, 'SATWRMID']
    score = train_model(features)
