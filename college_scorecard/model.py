import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.regularizers import l2
# from keras.utils import np_utils
import matplotlib.pyplot as plt
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


def get_data():
    data = {}

    cwd = os.getcwd()
    files = sorted(glob.glob(cwd + '/CollegeScorecard_Raw_Data/*.csv'))
    files = [cwd + '/CollegeScorecard_Raw_Data/MERGED2013_14_PP.csv']
    for file in files:
        df = pandas.read_csv(file, delimiter=',', error_bad_lines=False)
        year = file.split('/')[-1][6:-7]
        male_earnings = get_clean_data(df['MN_EARN_WNE_MALE1_P6'])
        male_count = get_clean_data(df['COUNT_WNE_MALE1_P6'])

        female_earnings = get_clean_data(df['MN_EARN_WNE_MALE0_P6'])
        female_count = get_clean_data(df['COUNT_WNE_MALE0_P6'])

        average_earnings = []
        for (me, mc, fe, fc) in zip(male_earnings, male_count, female_earnings, female_count):
            average_earnings.append((me * mc + fe * fc) / (mc + fc) if -1 not in (me, mc, fe, fc) else -1.0)
        average_earnings = np.asarray(average_earnings)
        print_overall_average(file, average_earnings, 'earnings')

        average_sat = get_clean_data(df['SAT_AVG_ALL'])
        print_overall_average(file, average_sat, 'SAT')

        dataset_df = pandas.DataFrame({'X': average_sat, 'Y': average_earnings})
        dataset_df_filtered = dataset_df.query('X>-1.0').query('Y>-1.0')
        dataset_filtered = dataset_df_filtered.values
        if dataset_filtered.size > 0:
            data[year] = dataset_filtered
    return data


dataset = get_data()
data = dataset['2013_14']
X, Y = data[:, 0], data[:, 1]

# Make test and train set
train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)

ax = sns.regplot(x='X', y='Y', data=pandas.DataFrame({'X': test_X, 'Y': test_y}))
ax.set(xlabel='Average SAT', ylabel='Mean earnings 6 years after entry')
plt.show()

model = Sequential()
model.add(Dense(1, input_shape=(1,), activation="linear"))

model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.01))
print(model.summary())

history = model.fit(train_X, train_y,
                    batch_size=16,
                    epochs=100,
                    verbose=1,
                    validation_data=(test_X, test_y))
score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss: %.4f' % score)
