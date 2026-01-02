This repository provides a comprehensive toolkit for epidemiological forecasting that combines Poisson regression with ARIMA time series modeling. The hybrid approach uses Socio-demographic Index (SDI) as a covariate in the Poisson regression component, while the ARIMA model captures residual temporal autocorrelation in the data. This repository consists of two main components:

ARIMA Parameter Finder-Automatically identifies optimal ARIMA (p,d,q) parameters for the residual time series.

Predictions-Performs integrated forecasting using:
Poisson regression with SDI covariate for baseline predictions
ARIMA modeling of residuals to capture temporal patterns


