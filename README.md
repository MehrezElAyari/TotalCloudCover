# README.md

## Abstract
Accurate and reliable forecasting of total cloud cover (TCC) is vital for many areas such as astronomy, energy demand and production, or agriculture. Most meteorological centres issue ensemble forecasts of TCC; however, these forecasts are often uncalibrated and exhibit worse forecast skill than ensemble forecasts of other weather variables. As TCC observations are usually reported on a discrete scale taking just nine different values called oktas, statistical calibration of TCC ensemble forecasts can be considered a classification problem with outputs given by the probabilities of the oktas. This is a classical area where machine learning methods are applied. In this [paper](https://doi.org/10.1007/s00521-020-05139-4), we investigate the performance of various post-processing techniques. This repository includes just the necessary code for the multilayer perceptron (MLP) neural networks technique.

## Data Used
The data used in this project includes 52 members of [ECMWF](https://www.ecmwf.int/) global ensemble forecasts:
- 1 high-resolution forecast (HRES)
- 1 control forecast (CTRL)
- 50 members generated using random perturbation (ENS)

The considered members are initialized at 1200 UTC for 10 lead times (1 day to 10 days) for the period from the 1st of January 2002 to the 20th of March 2014 for:
- Total Cloud Coverage (TCC) and the corresponding observations
- 24 h precipitation accumulation and the corresponding observations

### Data Details
- The Total Cloud Coverage (TCC) data set is for 3330 synoptic observation stations.
  - It’s a continuous quantity in the [0, 1] interval but it’s reported in values {0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1}
- The precipitation data set is for 2917 SYNOP stations.
- For only 2239 SYNOP stations both TCC and precipitation data are available.

## How to Use
To use the provided code, follow these steps:
1. Place the Python programs in the same directory as the SYNOPTIC station files.
2. Run the Python scripts. They will generate an `.RData` file for each station that contains the probabilities and different verification scores.

### Prerequisites
- Ensure you have Python installed on your system.
- Install the required Python libraries. You can do this using:
  ```bash
  pip install -r requirements.txt
