'''
========================================
Student model
========================================

Test loss: 49939391.2836
Features: ['SAT_AVG_ALL']
Number of examples: 9826

Test loss: 68742027.3115
Features: ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID', 'SATWRMID']
Number of examples: 3253

Test loss: 51394117.5950
Features: ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID']
Number of examples: 8015

Test loss: 56046866.9405
Features: ['SATVRMID', 'SATMTMID']
Number of examples: 8016

========================================
School model
========================================

Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 1)                 8
=================================================================
Total params: 8
Trainable params: 8
Non-trainable params: 0
_________________________________________________________________
None
Test loss: 62799035.3775
Features: ['ADM_RATE_ALL', 'AVGFACSAL', 'TUITIONFEE_IN', 'TUITIONFEE_OUT', 'PFTFAC', 'ICLEVEL', 'GRADS']
Number of examples: 8396
Batch size: 200, learning rate: 0.5000


=====================================================
Normalized student model

Coefficients:
 [[ 25184.13366366 -44190.50648594  54903.45210942]]
R squared: 0.4375
Features: ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID']
Number of examples: 8015


Normalized school model

Coefficients:
 [[ 0.01772076  0.51360059  0.50706065 -0.58785883 -0.09126908  0.23001001]]
R squared: 0.1199
Features: ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID', 'ADM_RATE_ALL', 'AVGFACSAL', 'TUITIONFEE_IN', 'TUITIONFEE_OUT', 'PFTFAC', 'GRADS']
Number of examples: 6536

'''

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
# from keras.regularizers import l2
# from keras.utils import np_utils
import matplotlib.pyplot as plt
# import seaborn as sns
import keras
import pandas
import numpy as np
import glob
import os
import argparse
from keras.models import model_from_json
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from collections import OrderedDict


def get_clean_data(column, datatype='float'):
    # Replace bad values with -1
    column.replace('PrivacySuppressed', -1, inplace=True)
    column.fillna(-1, inplace=True)
    values = column.values.astype(np.float) if datatype == 'float' else column.values
    return values


def print_overall_stats(year, data, name):
    data = data[data != -1.0]
    n_points = data.size
    if n_points == 0:
        print("Year: %s, Feature: %s, No data" % (year, name))
        return
    overall_aver = np.average(data)
    overall_max = np.amax(data)
    overall_min = np.amin(data)
    print("Year: %s, Feature: %s, Average: %s, Max %s, Min: %s, Data points: %d" %
          (year, name, overall_aver, overall_max, overall_min, n_points))


def print_death_rate(df, year):
    # Death rate
    dd = {'death_rate': get_clean_data(df['DEATH_YR2_RT']),
          'name': get_clean_data(df['INSTNM'], datatype='')}
    dd_df = pandas.DataFrame(dd)
    dd_df.sort_values('death_rate', ascending=False, inplace = True)
    print("Year: %s" % year)
    print("Death rate")
    print(dd_df.head(50))


def load_model(name, type):
    if type == 'keras':
        json_file = open('%s.json' % name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("%s.h5" % name)
        # need to compile, otherwise it returns nonsense (such as negative values)
        loaded_model.compile(loss='mse', optimizer='adam')
        model_features = 3
        print("Loaded student model from disk")
    elif type == 'sklearn':
        loaded_model = joblib.load('%s.pkl' % name)
        model_features = loaded_model.rank_
        print("Loaded student model from disk")
    else:
        loaded_model = None
    return loaded_model, model_features


def compute_average_earnings(dataframe):
    # Average earnings by male after 6 years of entry
    male_earnings = get_clean_data(dataframe['MN_EARN_WNE_MALE1_P6'])
    # Male count
    male_count = get_clean_data(dataframe['COUNT_WNE_MALE1_P6'])

    # Average earnings by male after 6 years of entry
    female_earnings = get_clean_data(dataframe['MN_EARN_WNE_MALE0_P6'])
    # Female count
    female_count = get_clean_data(dataframe['COUNT_WNE_MALE0_P6'])

    # Death rate
    # print_death_rate(df, year)

    # Average earnings (female and male)
    average_earnings = []
    for (me, mc, fe, fc) in zip(male_earnings, male_count, female_earnings, female_count):
        average_earnings.append((me * mc + fe * fc) / (mc + fc) if -1.0 not in (me, mc, fe, fc) else -1.0)
    # Convert list to numpy array
    average_earnings = np.asarray(average_earnings)
    return average_earnings


def get_data(features):
    data_filtered = {}
    data = OrderedDict({})

    cwd = os.getcwd()
    files = sorted(glob.glob(cwd + '/CollegeScorecard_Raw_Data/*.csv'))
    # files = [cwd + '/CollegeScorecard_Raw_Data/MERGED2013_14_PP.csv']
    for file in files:
        # Extract year from file name
        year = file.split('/')[-1][6:-7]

        # Consider only data from 2xxx (previous years don't have the data we are interested in)
        if year.startswith('1'):
            continue
        df = pandas.read_csv(file, delimiter=',', error_bad_lines=False)

        # Death rate
        # print_death_rate(df, year)

        for feature in features:
            # Get feature data
            feature_data = get_clean_data(df[feature])
            data[feature] = feature_data
            # Print overall feature average
            print_overall_stats(year, feature_data, feature)

        # Compute earnings
        average_earnings = compute_average_earnings(df)
        data['Y'] = average_earnings
        # Print overall earning average
        print_overall_stats(year, average_earnings, 'EARNINGS')

        # Convert to dataframe
        dataset_df = pandas.DataFrame(data)
        # Filter missing values
        for col in dataset_df.columns:
            query = '%s>-1.0' % col
            dataset_df = dataset_df.query(query)
        dataset_filtered = dataset_df.values

        # Skip years with no data
        if dataset_filtered.size > 0:
            data_filtered[year] = dataset_filtered
            print("Data after filtering")
            for c in range(dataset_filtered.shape[1]):
                feature_name = features[c] if c < len(features) else "EARNINGS"
                print_overall_stats(year, dataset_filtered[:, c], feature_name)

    return data_filtered


def train_model(features, model_name, batch=16, n_epochs=300, learning_rate=0.1, model_type='keras'):
    assert model_type in ('keras', 'sklearn'), "Model type should be keras or sklearn"
    n_features = len(features)

    dataset = get_data(features)
    data = None
    for key, value in dataset.items():
        if data is None:
            data = value
        else:
            data = np.concatenate((data, value), axis=0)

    # Normalize - min max normalization. Don't normalize earnings
    for c in range(data.shape[1] - 1):
        col = data[:, c]
        col_max, col_min = np.amax(col), np.amin(col)
        feature_name = features[c] if c < len(features) else "EARNINGS"
        print("Feature: %s, max: %.4f, min: %.4f" % (feature_name, col_max, col_min))
        data[:, c] = (col - col_min)/(col_max - col_min)

    loaded_model_n_features = 0
    # Load student model if building a school model
    if model_name == 'student':
        X, Y = data[:, :n_features], data[:, -1:]
    else:
        # Load student model
        loaded_model, loaded_model_n_features = load_model('student', model_type)
        assert loaded_model is not None, "Can't load a model"

        X_student_features = data[:, :loaded_model_n_features]
        X = data[:, loaded_model_n_features:n_features]
        Y_raw = data[:, -1:]

        Y_student_predicted = loaded_model.predict(X_student_features)
        Y = np.divide(Y_raw, Y_student_predicted)

    features_used = features[loaded_model_n_features:n_features]

    # Make test and train set
    train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)

    # Linear regression
    if model_type == 'keras':
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        model = Sequential()
        model.add(Dense(1, input_shape=(len(X[0]),), kernel_initializer='normal', activation="linear"))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))  #RMSprop(lr=0.05))

        # train
        history = model.fit(train_X, train_y,
                            batch_size=batch,
                            epochs=n_epochs,
                            verbose=1,
                            validation_data=(test_X, test_y))

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.draw()

        # Evaluate and print MSE
        score = model.evaluate(test_X, test_y, verbose=0)
        print("Test loss: %.4f" % score)

        # serialize model to JSON
        model_json = model.to_json()
        with open("%s.json" % model_name, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("%s.h5" % model_name)
        print("Saved model to disk")
    elif model_type == 'sklearn':
        model = linear_model.LinearRegression()
        model.fit(train_X, train_y)
        joblib.dump(model, '%s.pkl' % model_name)
        print("Saved model to disk")

    predictions = model.predict(test_X)
    r_squared = r2_score(test_y, predictions)

    if model_type == 'keras':
        print("Model weights: %s" % model.get_weights())
        print(model.summary())
        print("Batch size: %d, learning rate: %.4f" % (batch, learning_rate))

    elif model_type == 'sklearn':
        # The coefficients
        print('Coefficients: \n', model.coef_)

    print("R squared: %.4f" % r_squared)
    print("Features: %s" % features_used)
    print("Number of examples: %d" % len(Y))

    # # Regression plot
    # ax = sns.regplot(x='X', y='Y', data=pandas.DataFrame({'X': test_X, 'Y': test_y}))
    # ax.set(xlabel='Average SAT', ylabel='Mean earnings 6 years after entry')
    # plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model_type', type=str, default='keras')
    args = parser.parse_args()

    features_student = [
        'SAT_AVG_ALL',  # Average SAT
        'SATVRMID',
        'SATMTMID',
        # 'SATMT75',
        # 'SATVR75',
        # 'SATMT25',
        # 'SATVR25'
    ]

    features_college = [
        'ADM_RATE_ALL',  # admission rate
        'AVGFACSAL',  # faculty salary
        'TUITIONFEE_IN',  # In-state tuition and fees
        'TUITIONFEE_OUT',  # Out-of-state tuition and fees
        'PFTFAC',  # Proportion of faculty that is full-time
        # 'ICLEVEL',  # Level of institution
        'GRADS',  # Number of graduate students
        # 'CONTROL',  # Control of institution (public, private nonprofit, public for profit)
    ]

    if args.stage == 'student':
        # Student model
        train_model(features_student, model_name=args.stage, batch=16, n_epochs=args.epochs,
                    learning_rate=0.1, model_type=args.model_type)
    elif args.stage == 'school':
        # School model
        features = features_student + features_college
        train_model(features, model_name=args.stage, batch=200, n_epochs=args.epochs,
                    learning_rate=0.001, model_type=args.model_type)
    plt.show()
