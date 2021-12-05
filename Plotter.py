import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider, Button

class Plotter:
    def __init__(self, models):
        self.fig, self.ax = plt.subplots()
        self.models = models
        self.longitude = 6
        self.latitude = 51
        self.year = 2012
        self.model = models[0]
        self.temp_graph(None)

    def temp_graph(self, event):
        self.ax.cla()
        predictions = []
        for month in range(1, 13):
            predictions.append(((self.model[0].predict([[(self.year - self.model[1][0]) / self.model[2][0], (month - self.model[1][1]) / self.model[2][1],
                                            (self.latitude - self.model[1][4]) / self.model[2][4], (self.longitude - self.model[1][5]) / self.model[2][5]]]))*self.model[2][2] + self.model[1][2])[0])

        self.ax.bar(np.array(list(range(0, 12))) + 0.2, predictions, 0.6, color=['r' if p > 0 else 'b' for p in predictions])
        self.ax.set_xticks(np.array(list(range(0, 12))) + 0.5)
        self.ax.set_xticklabels(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        self.ax.set_xlabel("Month")
        self.ax.set_ylabel("°C")
        self.ax.plot()

    def lat(self, x):
        self.latitude = x
        self.temp_graph(None)

    def long(self, x):
        self.longitude = x
        self.temp_graph(None)

    def year_changed(self, x):
        self.year = x
        self.temp_graph(None)

    def model_changed(self, x):
        self.model = self.models[int(round(x, 0))]
        self.temp_graph(None)

    def start(self):
        plt.subplots_adjust(bottom=0.3)
        lat = plt.axes([0.25, 0.2, 0.5, 0.03])
        long = plt.axes([0.25, 0.16, 0.5, 0.03])
        year_choice = plt.axes([0.25, 0.12, 0.5, 0.03])
        model_choice = plt.axes([0.25, 0.08, 0.5, 0.03])

        self.box_lat = Slider(lat, 'Lat', -90, 90, 51, '%1.0f' ,valstep=1)
        self.box_lat.on_changed(self.lat)
        self.box_long = Slider(long, 'Lon',  -180, 180, 6, '%1.0f', valstep=1)
        self.box_long.on_changed(self.long)
        self.box_year = Slider(year_choice, 'Year', 1700, 3500, 2012, '%1.0f', valstep=1)
        self.box_year.on_changed(self.year_changed)
        self.box_model = Slider(model_choice, 'Model',  0, len(self.models) - 1, 0, '%1.0f', valstep=1)
        self.box_model.on_changed(self.model_changed)
        plt.show()
