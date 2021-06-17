import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn import tree
from scipy import stats


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]


# plot precision, recall vs threshold


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precisions[:-1],
            name="Precision",
            mode="lines",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recalls[:-1],
            name="Recall",
            mode="lines",
            line=dict(color="green"),
        )
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[-50000, 50000])
    fig.update_layout(
        title="Precision and recall versus the decision threshold",
        xaxis_title="Threshold",
    )
    fig.show()


def plot_precision_vs_recall(precisions, recalls):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=recalls, y=precisions, mode="lines", line=dict(color="green"))
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(
        title="Precision vs Recall", xaxis_title="Recall",
    )
    fig.show()


def plot_roc_curve(fpr, trp, label=None):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="green"), name=label)
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="blue"),
            name="random classifier",
        )
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    if label == None:
        fig.update_layout(
            title="The ROC Curve",
            xaxis_title="False Positive Rate (Fall-Out)",
            yaxis_title="True Positive Rate (Recall)",
            showlegend=False,
        )
    else:
        fig.update_layout(
            title="The ROC Curve",
            xaxis_title="False Positive Rate (Fall-Out)",
            yaxis_title="True Positive Rate (Recall)",
        )

    fig.show()


def compare_roc_curve(fpr_clf1, trp_clf1, label1, fpr_clf2, tpr_clf2, label2):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_clf1, y=trp_clf1, mode="lines", line=dict(color="green"), name=label1
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr_clf2, y=tpr_clf2, mode="lines", line=dict(color="red"), name=label2
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="blue"),
            name="random classifier",
        )
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(
        title="The ROC Curve",
        xaxis_title="False Positive Rate (Fall-Out)",
        yaxis_title="True Positive Rate (Recall)",
    )

    fig.show()


from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


def plot_learning_curves(estimator, X, y, cv):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        n_jobs=-1,
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean,
            name="Training accuracy",
            mode="lines",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_mean,
            name="Validation accuracy",
            mode="lines",
            line=dict(color="green"),
        )
    )

    fig.update_layout(
        title="Learning Curves",
        xaxis_title="Number of training examples",
        yaxis_title="Accuracy",
    )

    fig.show()


def plot_validation_curves(estimator, X, y, param_name, param_range, cv=3):
    train_scores, test_scores = validation_curve(
        estimator=estimator,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=train_mean,
            name="Training Accuracy",
            mode="lines",
            line=dict(color="Blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=test_mean,
            name="Validation Accuracy",
            mode="lines",
            line=dict(color="Green"),
        )
    )

    fig.update_layout(
        title="Validation Curves", xaxis_title=param_name, yaxis_title="Accuracy"
    )

    fig.show()


def plot_decision_tree(classifier, feature_names, class_names):
    """This function plots decision tree.
    classifier: The name of the classifier,
    feature_names: Feature names
    class_name: class names
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    tree.plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        filled=True,
    )
    fig.show()


def plot_silhouetter_scores(k_range, silhouette_scores):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=silhouette_scores,
            mode="lines+markers",
            marker=dict(color="green"),
        )
    )
    fig.update_layout(xaxis_title="K", yaxis_title="Silhouette Score")
    fig.show()


def num_to_cat_list(df, num_col_list, n_unique_val):
    """This function takes a pandas dataframe, a list of numerical columns
    and create a list of columns that needs to be converted to categorical column if
    it is less than or equal to n_unique_val."""

    # columns that needs to converted
    cols_to_convert = []
    for col in num_col_list:
        unique_val = df[col].nunique()
        print(col, unique_val)
        if unique_val <= n_unique_val:
            cols_to_convert.append(col)
    return cols_to_convert


def ci_mean_std_known(array, std, conf_level=95):
    """
    This function calculates the confidence interval for the population mean
    when standard deviation is known.
    
    array: python list or pandas series
    mean: mean of the sample
    std: standard deviation
    n: sample size
    conf_level: confidence level, default to 95 for 95% confidence interval
    alpha: significance level
    
    """
    # calculate significance level
    alpha = np.round((1 - conf_level / 100), 2)
    mean = np.mean(array)
    n = len(array)
    # calculate standard error
    std_error = std / np.sqrt(n)
    # find z critical value
    z_star = np.round(stats.norm.ppf(1 - alpha / 2), 3)
    # margin of error
    margin_of_error = np.round(z_star * std_error, 2)

    # calculate the lower and upper confidence bounds
    lcb = np.round(mean - margin_of_error, 2)
    ucb = np.round(mean + margin_of_error, 2)

    print("Margin Of Error: {}".format(margin_of_error))
    print(
        "{}% Confidence Interval for Population Mean: ({}, {})".format(
            conf_level, lcb, ucb
        )
    )


def ci_mean_std_unknown(array, conf_level=95):
    """
    This function calculates the confidence interval for a population mean
    when the standard deviation is unknown.
    
    array: pandas series or python list
    conf_level: confidence level, default to 95 for 95% confidence interval
    
    """
    # calculate significance level
    alpha = np.round((1 - conf_level / 100), 2)
    # mean of the sample
    mean = np.mean(array)
    # standard deviation
    std = np.std(array)
    # size of the sample
    n = len(array)
    # degrees of freedom
    df = n - 1
    # calculate the standard error
    std_error = std / np.sqrt(n)
    # find the t critical value
    t_star = np.round(stats.t.ppf(1 - alpha / 2, df), 3)
    # margin of error
    margin_of_error = np.round(t_star * std_error, 2)
    # calculate the lower and upper confidence bounds
    lcb = np.round(mean - margin_of_error, 2)
    ucb = np.round(mean + margin_of_error, 2)

    print("Margin Of Error: {}".format(margin_of_error))
    print(
        "{}% Confidence Interval for Population Mean: ({},{})".format(
            conf_level, lcb, ucb
        )
    )


def ci_diff_mean_std_known(array1, array2, std1, std2, conf_level=95):
    """
    This function calculates the Confidence Interval for the difference between two means.
    array1: sample one values as pandas series or python list
    aaray2: sample two values as pandas series or python list
    std1: standard deviation of sample 1
    std2:standard deviation of sample 2
    conf_level: confidence level, default to 95 for 95% confidence interval
    """
    # calculate significance level
    alpha = np.round((1 - conf_level / 100), 2)

    # means of samples
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)

    # size of the samples
    n1 = len(array1)
    n2 = len(array2)

    # difference of the two means
    diff_mean = mean1 - mean2

    # the z critical value
    z_star = np.round(stats.norm.ppf(1 - alpha / 2), 3)

    # margin of error
    margin_of_error = z_star * np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

    # upper and lower confidence bounds
    lcb = np.round(diff_mean - margin_of_error, 2)
    ucb = np.round(diff_mean + margin_of_error, 2)

    print(
        "{}% Confidence Interval for difference of two population means: ({},{})".format(
            conf_level, lcb, ucb
        )
    )


def deg_fred_two_means(s1, s2, n1, n2):
    """
    s1 : The sample one standard deviation
    s2 : The sample two standard deviation
    n1 : sample one size
    n2 : sample two size
    """
    num = ((s1 ** 2 / n1) + (s2 ** 2 / n2)) ** 2
    denom = (1 / (n1 - 1) * (s1 ** 2 / n1) ** 2) + (1 / (n2 - 1) * (s2 ** 2 / n2) ** 2)
    deg_fred = int(num / denom)
    return deg_fred


def ci_diff_mean_std_unknown(array1, array2, conf_level=95):
    """
    This function calculates the Confidence Interval for the difference between two means.
    array1: sample one values as pandas series or python list
    aaray2: sample two values as pandas series or python list
    conf_level: confidence level, default to 95 for 95% confidence interval
    """
    # calculate significance level
    alpha = np.round((1 - conf_level / 100), 2)

    # means of samples
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)

    # standard deviation fo samples
    std1 = np.std(array1)
    std2 = np.std(array2)

    # size of the samples
    n1 = len(array1)
    n2 = len(array2)

    # difference of the two means
    diff_mean = mean1 - mean2

    # degrees of freddom
    deg_fred = deg_fred_two_means(std1, std2, n1, n2)

    # find the t critical value
    t_star = np.round(stats.t.ppf(1 - alpha / 2, deg_fred), 3)

    # margin of error
    margin_of_error = t_star * np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

    # upper and lower confidence bounds
    lcb = np.round(diff_mean - margin_of_error, 2)
    ucb = np.round(diff_mean + margin_of_error, 2)

    print(
        "{}% Confidence Interval for difference of two population means: ({},{})".format(
            conf_level, lcb, ucb
        )
    )


def ci_prop(p, n, conf_level=95):
    """
    This function calculates the confidence interval for a population proportion.
    p: proportion for the sample
    n: sample size
    conf_level: confidence level, default to 95 for 95% confidence interval
    """
    # calculate significance level
    alpha = np.round((1 - conf_level / 100), 2)
    # standard error
    std_error = np.sqrt(p * (1 - p) / n)
    # find the z critical value
    z_star = np.round(stats.norm.ppf(1 - alpha / 2), 3)
    # margin of error
    margin_of_error = np.round(z_star * std_error, 2)
    # calculate lower and upper confidence bounds
    lcb = np.round(p - margin_of_error, 2)
    ucb = np.round(p + margin_of_error, 2)

    print("Margin Of Error: {}".format(margin_of_error))
    print(
        "{}% Confidence Interval for Population Proportion: ({}, {})".format(
            conf_level, lcb, ucb
        )
    )


def ci_diff_prop(p1, p2, n1, n2, conf_level=95):
    """
    This function calculates Confidence Interval for the difference in two population proportions. 
    p1: proportion of sample one
    p2: proportion of sample two
    n1: size of sample one
    n2: size of sample two
    conf_level: confidence level, default to 95 for 95% confidence interval
    """
    # calculate significance level
    alpha = np.round((1 - conf_level / 100), 2)
    prop_diff = p1 - p2
    # find the z critical value
    z_star = np.round(stats.norm.ppf(1 - alpha / 2), 3)
    margin_of_error = z_star * (np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)))
    # calculate the lower and upper bound
    lcb = prop_diff - margin_of_error
    ucb = prop_diff + margin_of_error
    print(
        "{}% Confidence Interval for difference in two Population proportions: ({},{})".format(
            conf_level, lcb, ucb
        )
    )


# function for describing date column
def describe_date(date):
    """
    This function takes a pandas date column and give some summary
    information between two dates.
    """
    min_date = date.min()
    max_date = date.max()
    total_months = (
        pd.to_datetime(date.max()).year - pd.to_datetime(date.min()).year
    ) * 12 + (pd.to_datetime(date.max()).month - pd.to_datetime(date.min()).month)
    total_days = str(pd.to_datetime(date.max()) - pd.to_datetime(date.min())).split(
        " "
    )[0]
    print("--------")
    print("Min date:", min_date)
    print("Max date:", max_date)
    print("Total Months:", total_months)
    print("Total Days:", total_days)
