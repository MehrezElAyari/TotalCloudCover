#!python
# encoding: utf-8
from datetime import datetime

import numpy as np
import rpy2.robjects as robjects
import tensorflow as tf
from keras import layers, models
from rpy2.robjects.numpy2ri import numpy2ri

zj = np.array([0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1])


class Station:
    def __init__(self, name):
        try:
            robjects.r["load"](name)
            if robjects.r["exists"]("cloud.data")[0] == 1:
                print("Loading data from Station : " + name.replace(".Rdata", ""))
                cld = robjects.r["cloud.data"]
                self.name = name
                self.metadata = cld
                self.rawdates = np.array(cld.rx2("dates"))
                self.dates = [
                    datetime.strptime(j, "%Y-%m-%d")
                    for j in [
                        i[0:4] + "-" + i[4:6] + "-" + i[6:8] for i in self.rawdates
                    ]
                ]
                self.lead_times = np.array(cld.rx2("leadtimes"))
                self.ensembles = np.array(cld.rx2("members"))
                a = np.array(cld.rx2("observation"))[:, 0]
                if not (all(np.in1d(a[~np.isnan(a)], zj))):
                    print("Correcting the Observations.")
                    for i in range(0, a.size):
                        if ~np.isnan(a[i]):
                            a[i] = zj[
                                np.searchsorted(
                                    [
                                        0.01,
                                        0.1875,
                                        0.3125,
                                        0.4375,
                                        0.5625,
                                        0.6875,
                                        0.8125,
                                        0.99,
                                        1,
                                    ],
                                    a[i],
                                    side="left",
                                )
                            ]
                    self.observations = a
                else:
                    self.observations = a
                self.idxnans = np.argwhere(np.isnan(self.observations))
                self.observations = np.delete(self.observations, self.idxnans)
                self.classes = sorted(
                    [i for i in list(set(self.observations)) if str(i) != "nan"]
                )
                self.forecasts = np.swapaxes(np.array(cld.rx2("forecasts")), 1, 2)
                self.forecasts = np.delete(self.forecasts, self.idxnans, axis=0)
                self.rawdates = list(np.delete(self.rawdates, self.idxnans, axis=0))
                for index in sorted(list(self.idxnans), reverse=True):
                    del self.dates[index[0]]
                self.StartYear = min(self.dates).year
                self.LastYear = max(self.dates).year
                self.factors = np.nan
            else:
                print("Cannot find variable " + name + " . Loading is stopping.")
                pass
        except:
            print("File not found")

    def Factors(self):
        if np.isnan(self.factors):
            robjects.r["load"](self.name)
            robjects.r("""
                                                res <- array(NA, c(length(cloud.data$dates), 7, 10))
                                                for (i in 1:length(cloud.data$leadtimes)){
                                                #r 
                                                res[,1,i]<-r <-  apply(cloud.data$forecasts[,i,1:50],1,mean,na.rm = T)
                                                #r_hres 
                                                res[,2,i]<-r_hres <-  cloud.data$forecasts[,i,51]
                                                #r_ctrl 
                                                res[,3,i]<-r_ctrl <-  cloud.data$forecasts[,i,52]
                                                #s2 
                                                res[,4,i] <- s2 <- apply(cloud.data$forecasts[,i,],1,var,na.rm = T)
                                                #f0 
                                                res[,5,i]<- apply(cloud.data$forecasts[,i,]==0,1,mean,na.rm = T)
                                                #f1 
                                                res[,6,i]<- apply(cloud.data$forecasts[,i,]==1,1,mean,na.rm = T)
                                                #i 
                                                res[,7,i]<- s2*sign((r+r_hres+r_ctrl)/3-0.5)*((r+r_hres+r_ctrl)/3-0.5)^2
                                                }
                                                """)
            self.factors = np.array(robjects.r["res"])
            self.factors = np.delete(self.factors, self.idxnans, axis=0)
        else:
            print("Object is empty.")


def myloss(a, b):
    zj0 = zj.reshape(1, len(zj))
    zdiff = np.absolute(np.subtract(zj0, np.transpose(zj0)))
    zdiff = tf.constant(zdiff, shape=[zj0.shape[1], zj0.shape[1]])
    zdiff = tf.cast(zdiff, tf.float32)
    zj0 = tf.cast(zj0, tf.float32)

    a1 = tf.matmul(a, tf.transpose(zj0))
    c = tf.reduce_mean(
        tf.subtract(
            tf.reduce_sum(tf.multiply(b, tf.math.abs(a1 - zj0)), 1),
            tf.reduce_sum(tf.multiply(b, tf.matmul(b, zdiff)), 1) / 2,
        )
    )
    return c


def crps(P, obs):
    Zj_prime = np.array([abs(zj[i] - zj) for i in range(0, zj.shape[0])])
    c = np.zeros(shape=(P.shape[0], 10))
    for j in range(0, 10):
        c[:, j] = np.sum(
            np.multiply(
                P[:, :, j], np.array([abs(obs[i] - zj) for i in range(0, obs.shape[0])])
            ),
            axis=1,
        ) - 0.5 * np.sum(np.multiply(np.dot(P[:, :, j], Zj_prime), P[:, :, j]), axis=1)
    return c


#####################################################
if __name__ == "__main__":
    pass
