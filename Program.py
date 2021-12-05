from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import gaussian_process
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.gaussian_process import kernels
import time
from Plotter import Plotter


weather_data = []
# Download data here: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
def get_weather_data(nrows=None, sample=30000):
    global weather_data, mean, var
    data = pandas.read_csv("GlobalLandTemperaturesByCity.csv", delimiter=',', nrows=nrows, usecols=[0,1,2,5,6], encoding='utf8')
    data['AverageTemperature'].replace('', np.nan, inplace=True)
    data.dropna(subset=['AverageTemperature'], inplace=True)
    data = data.reset_index().values
    data = np.array(random.sample(list(data), sample))
    data[:, 0] = [int(date[0:4]) for date in data[:, 1]]
    data[:, 1] = [int(date[5:7]) for date in data[:, 1]]
    data[:, 4] = [float(lat.split('N')[0]) if 'N' in lat else -float(lat.split('S')[0]) for lat in data[:, 4]]
    data[:, 5] = [float(long.split('E')[0]) if 'E' in long else -float(long.split('W')[0]) for long in data[:, 5]]

    weather_data = data


models = []
def ModelStresstest(model, sample_size = None, test_ratio = 1/3, include_var = False):
    if sample_size is None:
        sample_size = len(weather_data)
    training_size = round(sample_size * (1 - test_ratio))
    testing_size = sample_size - training_size

    print("\n", training_size, model)

    data = np.copy(weather_data)
    mean = data[:training_size].mean(axis=0)

    # Gaussian Process Regression can take noise into account, do it
    include_var = include_var and "GaussianProcess" in str(model)

    # Standardize data for better processability
    if include_var:
        data[:, :3], data[:, 4:] = data[:, :3] - mean[:3], data[:, 4:] - mean[4:]
    else:
        data -= mean

    var = data[:training_size].var(axis=0)

    if include_var:
        data[:, :3], data[:, 4:], data[:, 3:4] = data[:, :3] / var[:3], data[:, 4:] / var[4:], data[:, 3:4] / 1.96 / var[2]
    else:
        data /= var

    if include_var:
        model.alpha = np.array(data[:training_size, 3], dtype=float)

    # Create training (sample) and testing (test_size) datasets
    X = np.c_[data[:training_size, :2], data[:training_size, 4:]]
    X_test = np.c_[data[-testing_size:, :2], data[-testing_size:, 4:]]
    y = data[:training_size, 2]
    y_test = data[-testing_size:, 2]

    # Train model on training data
    a = time.time()
    model.fit(X, y)
    print(time.time() - a, "seconds to fit", training_size, "samples")

    # Test model on testing data (out of sample error)
    a = time.time()
    prediction = model.predict(X_test) * var[2] + mean[2]
    print(time.time() - a, "seconds to predict")
    error_out = abs(prediction - (y_test * var[2] + mean[2]))
    mean_error = sum(abs(error_out)) / testing_size
    median = sorted(error_out)[int(len(error_out)/2)]
    print("e_out=", mean_error, median)
    
    # Test model on training data (in sample error)
    a = time.time()
    prediction = model.predict(X)
    error_in = abs((prediction * var[2] + mean[2]) - (y * var[2] + mean[2]))
    mean_error_in = sum(abs(error_in)) / training_size
    median = sorted(error_in)[int(len(error_in) / 2)]
    print("e_in=", mean_error_in, median)

    models.append([model, mean, var, mean_error, mean_error_in])
    return model, mean, var, mean_error, mean_error_in


sample_size = 10000
test_ratio = 1/3

plt.rcParams.update({'font.size': 15})
get_weather_data(sample=sample_size)
models_test = []

# Gaussian Process Regression
# length scale determines the importance of each feature (big = unimportant)
# Values determined through 
for length_scale in [[33.5, 0.315, 0.0382, 0.0115]]:
    for kernel in [kernels.RBF(length_scale)]:
        models_test.append([gaussian_process.GaussianProcessRegressor(kernel=0.0988**2 * kernel, optimizer=None), sample_size])

# Support Vector Machine (Regression)
for kernel in [('rbf', 2)]:
    models_test.append([SVR(kernel=kernel[0], degree=kernel[1], epsilon=0.01, C=1, gamma='scale'), sample_size])

# Random Forest (Regression)
for trees in [100]:
    models_test.append([RandomForestRegressor(n_estimators=trees), sample_size])

# Train models
for model in models_test:
    ModelStresstest(model[0], model[1], test_ratio, include_var=True)

# Sort by out of sample error
print(sorted(models, key=lambda x: x[3])[0])

# Interactive prediction UI
test = Plotter(models)
test.start()