#!python
# encoding: utf-8


"""
Total cloud coverage project (non seasonal)
12-07-2019
Mehrez El Ayari
"""

import glob
from datetime import datetime
from itertools import compress

import numpy as np
import rpy2.robjects as robjects
from keras import layers, models
from rpy2.robjects.numpy2ri import numpy2ri
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from my_module import Station, crps, myloss

#####################################################
if __name__ == "__main__":
    # Training Size
    TrainingSize = 5
    # MLP Classifier: setting the model
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(9, activation="softmax"))
    model.compile(optimizer="adam", loss=myloss, metrics=["accuracy"])

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    # getting the file list
    Stations = [file for file in glob.glob("*.rdata")]
    Stations = [f for f in Stations if not ("prob" in f)]
    # looping through all of the files
    for st in Stations:
        Data = Station(st)
        # First year
        StartYear = Data.StartYear
        # Last year
        LastYear = Data.LastYear
        # getting factors (in here the factors are the raw ensemble)
        factors = Data.forecasts
        # Data.Factors()
        # factors = Data.factors
        # OneHot transformation of the observations
        observations = label_encoder.fit_transform(Data.observations)
        observations = onehot_encoder.fit_transform(
            observations.reshape(len(observations), 1)
        )
        idx_pred_total = [l.year > StartYear + TrainingSize for l in Data.dates]
        ####################################
        pred_dates = list(compress(Data.rawdates, idx_pred_total))
        firstIterationyr = True
        for i in range(StartYear + TrainingSize, LastYear):
            idx = [
                ((l.year >= (i - TrainingSize)) and (l.year <= i)) for l in Data.dates
            ]
            idx_pred = [k.year == (i + 1) for k in Data.dates]
            obs_train = observations[idx, :]
            obs_pred = observations[idx_pred, :]
            firstIterationlt = True
            for lt in range(10):
                f_train = imp.fit_transform(factors[idx, :, lt])
                f_pred = imp.fit_transform(factors[idx_pred, :, lt])
                hist = model.fit(
                    f_train, obs_train, epochs=20, batch_size=64, verbose=0
                )
                # prob = model.predict_proba(f_pred)

                if firstIterationlt:
                    prob = model.predict_proba(f_pred)
                    firstIterationlt = False
                else:
                    prob = np.dstack((prob, model.predict_proba(f_pred)))
            if firstIterationyr:
                probf = prob
                firstIterationyr = False
            else:
                probf = np.row_stack((probf, prob))

        rdata = numpy2ri(probf)
        robs = numpy2ri(np.array(list(compress(Data.observations, idx_pred_total))))
        rdates = numpy2ri(np.array(pred_dates))
        rcrps = numpy2ri(
            crps(probf, np.array(list(compress(Data.observations, idx_pred_total))))
        )
        robjects.r.assign("prob", rdata)
        robjects.r.assign("dates", rdates)
        robjects.r.assign("obs", robs)
        robjects.r.assign("crps", rcrps)
        robjects.r("save(prob,dates,obs,crps, file='{}')".format("prob_" + Data.name))
        del Data
