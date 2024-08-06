# Spatial-prediction-of-groundwater-quality-in-Guanzhong-Plain
- [LightGBM.py](https://github.com/BD-Sengoku/Spatial-prediction-of-groundwater-quality-in-Guanzhong-Plain/blob/main/LightGBM.py)

Use TPE to tune the LightGBM model, iterating 1000 times to obtain the optimal model and save the related parameters.



- [MakeImageAboutShap.py](https://github.com/BD-Sengoku/Spatial-prediction-of-groundwater-quality-in-Guanzhong-Plain/blob/main/MakeImageAboutShap.py)

Based on SHAP analysis, plot the resulting model.



- [ForecastPointDataForTheWholeRegion.py](https://github.com/BD-Sengoku/Spatial-prediction-of-groundwater-quality-in-Guanzhong-Plain/blob/main/ForecastPointDataForTheWholeRegion.py)

Using the indicator data of all regional points, predict their groundwater quality with the resulting model.



- [PointDataForTheWholeRegion.csv](https://github.com/BD-Sengoku/Spatial-prediction-of-groundwater-quality-in-Guanzhong-Plain/blob/main/PointDataForTheWholeRegion.csv)

This is the dataset used to predict the groundwater quality of the entire region. It should be noted that the study area is large, and the original dataset is about 1 GB, making it difficult to upload. Therefore, only the first ten rows of data are shown.
