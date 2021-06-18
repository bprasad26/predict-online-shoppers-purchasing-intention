import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def prepare_data(train, test):
    """
    This function takes training and test dataframe
    and do all the data cleaning and preprocessing 
    for machine learning and 
    Return: (X_train, y_train, X_test, full_pipe)
    """
    # change the name of the target column
    train.rename(columns={"revenue": "target"}, inplace=True)
    # map bool values to yes and no
    train["Weekend"] = train["Weekend"].map({True: "Yes", False: "No"})
    test["Weekend"] = test["Weekend"].map({True: "Yes", False: "No"})
    # set the id col as index
    train.set_index("id", inplace=True)
    test.set_index("id", inplace=True)

    # seperate the fetures and the target
    X_train = train.drop("target", axis=1).copy()
    y_train = train["target"].copy()
    X_test = test.copy()

    # select numerical and categorical columns
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    # numerical pipeline
    num_pipe = make_pipeline(SimpleImputer(strategy="mean"))

    # categorical pipeline
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    # full pipeline for data preprocessing
    full_pipe = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )
    return X_train, y_train, X_test, full_pipe

