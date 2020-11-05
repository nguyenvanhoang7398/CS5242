# CS5242: Neural Networks and Deep Learning
This is a single-class multi-classification task based on medical images


## Resources
* Download and unzip data from [Kaggle](https://www.kaggle.com/c/nus-cs5242/data).
```
kaggle competitions download -c nus-cs5242; unzip nus-cs5242.zip
```

## Run project
Split model to 10 folds for cross-validation, train and generate predict to `predictions.csv`
```
python run_project.py
```

## Dependencies
sklearn==0.23.1
numpy==1.18.5
torch==1.7.0
tensorboard==2.3.0
