# Data Preparation

- Get the mapping file:

```
python get_info.py --folder_path /YOUR_FOLDER_PATH/Eprime/
```

You will get a folder named "info", which contains 3 mapping files "sample2video.xlsx", "video2sample.xlsx", "video2order.xlsx".

- Prepare the training data:

```
python preprocess.py --data_path /YOUR_FOLDER_PATH/output/ --feature_path /YOUR_FOLDER_PATH/virtual_feature/
```

You will get 2 folders named "X" and "y" for fitting models.


# Model Fitting

- Fit the full model:

```
bash run.sh -t full
```

- Fit the residual model:

```
bash run.sh -t res
```

All the results are recorded in "results" folder.

# Result Organization

- Organize all metrics:

```
python metric.py --metric all --model full_full
```

- Organize a specific metric:

```
python metric.py --metric 'Pearson Correlation Coefficient' --model full_full
```

All the metrics are recorded in "metrics" folder.
