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
=====================================================

SKlearn models:
=====================================================
Normalized student model

Coefficients:
 [[ 25184.13366366 -44190.50648594  54903.45210942]]
R squared: 0.4375
Features: ['SAT_AVG_ALL', 'SATVRMID', 'SATMTMID']
Number of examples: 8015


Normalized school model

Coefficients:
 [[ 0.01450093  0.50687372  0.50111331 -0.58039551 -0.09158081  0.219488  ]]
R squared: 0.1391
Features: ['ADM_RATE_ALL', 'AVGFACSAL', 'TUITIONFEE_IN', 'TUITIONFEE_OUT', 'PFTFAC', 'GRADS']
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


# Set seed for reproducibility
SEED = 7
np.random.seed(SEED)


class Model(object):
    def __init__(self, model_type, model_name, input_shape=None, params=None):
        assert model_type in ('sklearn', 'keras')
        self._type = model_type
        self._input_shape = input_shape
        self._model_name = model_name
        self._model = None

        self._params = params

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def type(self):
        return self._type

    def create(self):
        if self._type == 'keras':
            model = Sequential()
            model.add(Dense(1, input_shape=(self._input_shape,), kernel_initializer='normal', activation="linear"))
            model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self._params['learning_rate']))  # RMSprop(lr=0.05))
            self._model = model
        elif self._type == 'sklearn':
            model = linear_model.LinearRegression()
            self._model = model

    def load(self):
        if self._type == 'keras':
            json_file = open('%s.json' % self._model_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("%s.h5" % self._model_name)
            # need to compile, otherwise it returns nonsense (such as negative values)
            loaded_model.compile(loss='mse', optimizer='adam')
            self._input_shape = 3
        elif self._type == 'sklearn':
            self._model = joblib.load('%s.pkl' % self._model_name)
            self._input_shape = self._model.rank_
        print("Loaded %s model from disk" % self._model_name)

    def train(self, train_X, train_y, test_X, test_y):
        if self._type == 'keras':
            history = self._model.fit(train_X, train_y,
                                      batch_size=self._params['batch'],
                                      epochs=self._params['epochs'],
                                      verbose=1,
                                      validation_data=(test_X, test_y)
                                      )

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.draw()

            # Evaluate and print MSE
            score = self._model.evaluate(test_X, test_y, verbose=0)
            print("Test loss: %.4f" % score)
        elif self._type == 'sklearn':
            self._model.fit(train_X, train_y)
        print("%s model trained" % self._model_name)

    def save(self):
        if self._type == 'keras':
            # serialize model to JSON
            model_json = self._model.to_json()
            with open("%s.json" % self._model_name, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self._model.save_weights("%s.h5" % self._model_name)
        elif self._type == 'sklearn':
            joblib.dump(self._model, '%s.pkl' % self._model_name)
        print("Saved model to disk")

    def predict(self, x):
        return self._model.predict(x)

    def print(self):
        print("Model parameters:")
        if self._type == 'keras':
            print("Model weights: %s" % self._model.get_weights())
            print(self._model.summary())
            print("Params: %s" % self._params)
        elif self._type == 'sklearn':
            print('Coefficients: \n', self._model.coef_)



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


def df_feature_into_onehot(df, feature_name):
    # Get index of feature
    one_hot_index = df.columns.get_loc(feature_name)
    # Split dataframe into 3
    dfs = np.split(df, [one_hot_index, one_hot_index + 1], axis=1)
    # Transform feature into str type
    f = dfs[1].astype(np.int8).astype(str)
    # Transform feature into onehot
    f_onehot = pandas.get_dummies(f, prefix='', prefix_sep='')
    return pandas.concat([dfs[0], f_onehot, dfs[2]], axis=1), len(f_onehot.columns)


def get_data(features):
    data_filtered = {}
    # We need ordered dict to keep order of features
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

        # Compute earnings
        average_earnings = compute_average_earnings(df)
        data['Y'] = average_earnings
        # Print overall earning average
        print_overall_stats(year, average_earnings, 'EARNINGS')

        for feature in features:
            # Get feature data
            feature_data = get_clean_data(df[feature])
            data[feature] = feature_data
            # Print overall feature average
            print_overall_stats(year, feature_data, feature)

        # Convert to dataframe
        dataset_df = pandas.DataFrame(data)
        # Filter missing values
        for col in dataset_df.columns:
            query = '%s>-1.0' % col
            dataset_df = dataset_df.query(query)

        dataset_df, one_hot_width = df_feature_into_onehot(dataset_df, 'CONTROL')

        dataset_filtered = dataset_df.values

        # Skip years with no data
        if dataset_filtered.size > 0:
            data_filtered[year] = dataset_filtered
            print("Data after filtering")
            for c in range(1, dataset_filtered.shape[1]):
                feature_name = features[c - 1] if c < len(features) + 1 else features[-1]
                print_overall_stats(year, dataset_filtered[:, c], feature_name)
    return data_filtered


def train_model(features, model_name, batch=16, n_epochs=300, learning_rate=0.1, model_type='keras'):
    assert model_type in ('keras', 'sklearn'), "Model type should be keras or sklearn"
    params = {'batch': batch, 'epochs': n_epochs, 'learning_rate': learning_rate}
    n_features = len(features)

    # Merge data over multiple years
    print("Reading data...")
    dataset = get_data(features)
    data = None
    for key, value in dataset.items():
        if data is None:
            data = value
        else:
            data = np.concatenate((data, value), axis=0)

    # Normalize - min max normalization. Don't normalize earnings
    print("Normalizing...")
    for c in range(1, data.shape[1]):
        col = data[:, c]
        col_max, col_min = np.amax(col), np.amin(col)
        feature_name = features[c - 1] if c < len(features) + 1 else features[-1]
        print("Feature: %s, max: %.4f, min: %.4f" % (feature_name, col_max, col_min))
        data[:, c] = (col - col_min)/(col_max - col_min)

    loaded_model_n_features = 0
    # Load student model if we are building a school model
    if model_name == 'student':
        X, Y = data[:, 1:], data[:, 0]
    else:
        # Load student model
        print("Loading student model...")
        student_model = Model(model_type=model_type, model_name='student')
        student_model.load()
        loaded_model_n_features = student_model.input_shape
        assert student_model is not None, "Can't load a model"

        X_student_features = data[:, 1:loaded_model_n_features + 1]
        X = data[:, loaded_model_n_features + 1:]
        Y_raw = data[:, 0]

        Y_student_predicted = student_model.predict(X_student_features)
        Y = np.divide(Y_raw, Y_student_predicted)

    features_used = features[loaded_model_n_features:]

    # Make test and train set
    train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)

    print("Training %s model..." % model_name)
    model = Model(model_type=model_type, model_name=model_name, input_shape=X.shape[1], params=params)
    model.create()
    model.train(train_X, train_y, test_X, test_y)
    model.save()

    predictions = model.predict(test_X)
    r_squared = r2_score(test_y, predictions)

    model.print()

    print("R squared: %.4f" % r_squared)
    print("Features: %s" % features_used)
    print("Number of examples: %d" % len(Y))

    # # Regression plot
    # ax = sns.regplot(x='X', y='Y', data=pandas.DataFrame({'X': test_X, 'Y': test_y}))
    # ax.set(xlabel='Average SAT', ylabel='Mean earnings 6 years after entry')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='sklearn')
    parser.add_argument('--stage', type=str, default='all')
    parser.add_argument('--epochs', type=int, default=300)
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
        'CONTROL',  # Control of institution (public, private nonprofit, public for profit)
    ]

    if args.stage in ('student', 'all'):
        # Student model
        train_model(features_student, model_name='student', batch=16, n_epochs=args.epochs,
                    learning_rate=0.1, model_type=args.model_type)
    if args.stage in ('school', 'all'):
        # School model
        features = features_student + features_college
        train_model(features, model_name='school', batch=200, n_epochs=args.epochs,
                    learning_rate=0.001, model_type=args.model_type)
    plt.show()
