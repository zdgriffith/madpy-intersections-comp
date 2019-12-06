from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import keras

def read_data(city='Chicago'):
    """
    This function will read in the desired City and create new veriables
    like TurnDegree, ExitStreetId and so forth
    """
    df = pd.read_csv('train.csv')
    df = df[df.City==city]

    # I will be using this dictionary to convert direction to degrees
    degrees = {'N':0, 'NE':45, 'E':90, 'SE':135, 'S':180, 'SW':225, 'W':270, 'NW':315}

    df["EntryHeading_deg"] = df.EntryHeading.apply(lambda x:degrees[x])
    df["ExitHeading_deg"] = df.ExitHeading.apply(lambda x:degrees[x])
    df["TurnDegree"] = (df.EntryHeading_deg-df.ExitHeading_deg).apply(lambda x: x if abs(x) <=180 else (x+360 if x<0 else x-360))
    df["TurnDegre"] = df.TurnDegree.apply(lambda x: x if x != -180 else x*-1)

    # Lets assign a number(StreetId) to each street
    all_streets = np.concatenate([df.ExitStreetName.reindex().values, df.EntryStreetName.reindex().values])
    # there are some nan values so lets just replace them with Unknown
    street_name_list = ['Unknown' if type(x)==type(0.0) else x for x in all_streets]
    street_names = {name: num for num, name in enumerate(street_name_list)}
    df["EntryStreetId"] = np.array([street_names[x] if x in street_names else -999 for x in df.EntryStreetName])
    df["ExitStreetId"] = np.array([street_names[x] if x in street_names else -999 for x in df.ExitStreetName])

    # we also want to categorize the street by its type (road, boulevard, ...)
    street_types = {n: i for i, n in enumerate(np.unique([x.split()[-1] for x in street_names.keys()]))}
    street_name_to_type = {}
    for name in street_names.keys():
        typ = name.split()[-1]
        street_name_to_type[name] = street_types[typ]
    df["EntryStreetType"] = np.array([street_name_to_type[x] if x in street_names else -999 for x in df.EntryStreetName])
    df["ExitStreetType"] = np.array([street_name_to_type[x] if x in street_names else -999 for x in df.ExitStreetName])

    df["EnterHighway"] = np.array([1 if type(x)==type('') and x.split()[-1] in ['Broadway', 'Parkway', 'Expressway', 'Highway'] else 0 for x in df.EntryStreetName])
    df["ExitHighway"] = np.array([1 if type(x)==type('') and x.split()[-1] in ['Broadway', 'Parkway', 'Expressway', 'Highway'] else 0 for x in df.ExitStreetName])
    df['Season'] = np.array([1 if month in (12,1,2) else 2 if month in (6,7,8) else 3 for month in df.Month.reindex().values])
    df['RushHour'] = np.array([1 if hour in (7,8,9) else 2 if hour in (16,17,18) else 3 if hour>=10 and hour<=15 else 4 for hour in df.Hour])
    return df

def create_train_test(df, cols):
    """
    This will create a training/testing dataset.  Because neural nets want data
    between 0-1 this will scale based on min/max values for each var and return
    the scaler so that we can flip back and forth
    """
    data = []
    for col in cols:
        data.append(df[col])
    data = np.transpose(data)

    x = data[:,:-1]
    y = np.reshape(data[:,-1], (-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(x)
    xscale=scaler_x.transform(x)
    scaler_y.fit(y)
    yscale=scaler_y.transform(y)
    x_train, x_test, y_train, y_test = train_test_split(xscale, yscale)
    return x_train, x_test, y_train, y_test, scaler_x, scaler_y

def gradient_boost_regressor(x_train, y_train, x_test, scaler_x, yscaler_y, n_est=100, depth=3, lr=0.1):
    params = {'n_estimators': n_est, 'max_depth': depth, 'min_samples_split': 2,
            'learning_rate': lr, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(scaler_x.inverse_transform(x_train),
            scaler_y.inverse_transform(y_train))
    return np.reshape(clf.predict(scaler_x.inverse_transform(x_test)), (-1,1))

def neural_net(x_train, y_train, x_test, scaler_y, epoch=10,
        initializer='lecun_normal', activation='relu', layers=5):
    model = Sequential()
    # no reason for layer spaceing here, just picked 8 out of a hat
    layer_spacing = 8
    model.add(Dense(layers*layer_spacing, activation=activation, input_dim=np.shape(x_train)[1], kernel_initializer=initializer))
    for j in range(layers-1, 1, -1):
        model.add(Dense(j*layer_spacing, activation='relu'))
    model.add(Dense( np.shape(y_train)[1], activation='linear'))
    model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])
    output = model.fit(x_train, y_train, epochs=10)
    ynew = model.predict(x_test)
    return scaler_y.inverse_transform(ynew)

if __name__ == "__main__":
    df = read_data('Chicago')

    cols = ['Latitude', 'Longitude', 'Season', 'RushHour', 'Weekend',
                  'EntryHeading_deg', 'ExitHeading_deg', 'TurnDegree',
                  'EnterHighway', 'ExitHighway',
                  'TimeFromFirstStop_p80']
    x_train, x_test, y_train, y_test, scaler_x, scaler_y = create_train_test(df, cols)

    # lets  convert the test result back to unscaled for analysis
    target = scaler_y.inverse_transform(y_test)

    # sort the target points so they are in order which will make determining
    # our fit of the data easier
    ind = np.argsort(np.concatenate(target))

    plt.plot(target[ind],'b.')

    # for the predictions i'm going to fit a line to them so that its easier to
    # visualize the difference between different models:
    xp = np.linspace(0,len(target),100)

    target_gbr = gradient_boost_regressor(x_train, y_train, x_test, scaler_x, scaler_y, 100, 3, 0.1)
    p = np.poly1d( np.polyfit(range(0,len(target)), np.concatenate(target_gbr[ind]), 10))
    plt.plot(xp, p(xp), 'g-.')

    target_gbr_1000_5_01 = gradient_boost_regressor(x_train, y_train, x_test, scaler_x, scaler_y, 1000, 5, 0.1)
    p = np.poly1d( np.polyfit(range(0,len(target)), np.concatenate(target_gbr_1000_5_01[ind]), 10))
    plt.plot(xp, p(xp), 'y-.')

    target_nn_e10_relu_l5 = neural_net(x_train, y_train, x_test, scaler_y, epoch=10, activation='relu', layers=5)
    p = np.poly1d( np.polyfit(range(0,len(target)), np.concatenate(target_nn_e10_relu_l5[ind]), 10))
    plt.plot(xp, p(xp), 'm--')

    target_nn_e15_l3_relu =  neural_net(x_train, y_train, x_test, scaler_y, epoch=10, layers=3, activation='relu')
    p = np.poly1d( np.polyfit(range(0,len(target)), np.concatenate(target_nn_e15_l3_relu[ind]), 10))
    plt.plot(xp, p(xp), 'r--')

    target_nn_e15_l9 =  neural_net(x_train, y_train, x_test, scaler_y, epoch=15, layers=9, activation='relu')
    p = np.poly1d( np.polyfit(range(0,len(target)), np.concatenate(target_nn_e15_l9[ind]), 10))
    plt.plot(xp, p(xp), 'k--')

    plt.setp(plt.gca(), 'ylim',[0,100],'xlim',[0,len(ind)])
    plt.grid('on')
    plt.ylabel('TimeFromFirstStop_p80')
    plt.xlabel('Trainined on:  Latitutde, Longitude, Season, RushHour, Weekend,\nEntryHeading_deg, ExitHeading_deg, TurnDegree, EnterHighway, ExitHighway')
    plt.legend(['truth','GBR: est=100  depth=3, lr=0.1','GBR: est=1000 depth=5 lr=0.1','NN: epoch=10, layers=5, relu','NN: epoch=15 layers=3 relu','NN: epoch=15, layers=9, relu'])
    plt.title('Truth vs Predictions')
    plt.savefig('first_attempt.png')
