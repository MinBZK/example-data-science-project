import sys
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate
from pycaret.classification import ClassificationExperiment
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.download_dataset import download_from_kaggle

np.random.seed(69)


def exploratory_data_analysis():
    """
    This function will do the exploratory data analysis (EDA) on the recruitment dataset of the University of Utrecht.
    In most projects this is done in a Jupyter Notebook as this makes interaction easier, but a slice of the EDA is
    still implementated to give you an idea.
    :return: pd.Dataframe containing the data
    """
    datapath = "./data/recruitmentdataset-2022-1.3.csv"
    download_from_kaggle(datapath)

    df = pd.read_csv(datapath, index_col="Id")

    df.head()  # The data is about whether based on some characteristic people are hired or not
    df.shape  # There are 4000 datapoints with 13 feature columns and 1 label column
    df[df.duplicated()]  # There are 39 duplicate values, but this is to be expected based on the categorical features
    df.isna().sum()  # There are no missing values
    df.describe()  # There are just 3 numerical features, which are the age, university grade, and languages.
    cat_cols = [
        "gender",
        "nationality",
        "sport",
        "ind-debateclub",
        "ind-programming_exp",
        "ind-international_exp",
        "ind-entrepeneur_exp",
        "ind-exact_study",
        "company",
        "decision",
    ]

    # TODO: Bias across different companies
    # TODO: Say something about distribution of the target variable
    # TODO: plots of target variable to the sensitive features
    # TODO: Better explanation of splitting of the dataset

    for col in cat_cols:
        categories = df.groupby(col).size()
        # print(categories)
    # From the categorical features we see that the non-binary gender is underrepresented although it is from the true
    # distribution of society at this moment. The same holds for nationality where Dutch is overrepresented. almost 3/4
    # of the applicants got rejected which we need to take into account in our model as well.
    return df


def preprocessing_data(data: pd.DataFrame) -> (Pipeline, pd.DataFrame):
    """
    Preprocessing of the dataset, on the one hand for the AIF360 we need to parse the dataset to a StandardDataset
    which is an object needed to be able to use the features of AIF360. Also for the machine learning models it is very
    beneficial to transform the features, like one-hot encodings for the categorical variables or scaling of numerical
    features, like stated in this book [Machine Learning Design Patterns](https://g.co/kgs/tQhwiL5).
    :param data: pandas Dataframe as the input data
    :return: (preprocessor_pipeline, preprocessed_data ): Tuple of a Sklearn Pipeline to process the dataset and
    the preprocessed data
    """
    preprocessor_pipeline = ColumnTransformer(
        [
            ("numerical", StandardScaler(), ["age", "ind-university_grade", "ind-languages"]),
            (
                "categorical",
                OneHotEncoder(sparse_output=False),
                [
                    "gender",
                    "nationality",
                    "sport",
                    "ind-degree",
                    "company",
                    "ind-debateclub",
                    "ind-programming_exp",
                    "ind-international_exp",
                    "ind-exact_study",
                    "ind-entrepeneur_exp",
                ],
            ),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor_pipeline.set_output(transform="pandas")
    preprocessed_data = preprocessor_pipeline.fit_transform(data)
    return preprocessor_pipeline, preprocessed_data


def training_model(
    preprocessing_pipeline: Pipeline,
    x_data: pd.DataFrame,
    y_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    exploratory_model_analysis: bool = False,
) -> int:
    """
    This function will train a model on the dataframe provided
    :param preprocessing_pipeline: sklearn pipeline with the preprocessing steps for the dataset
    :param x_data: pandas Dataframe as the input data but only the features
    :param y_data: pandas Dataframe as the input data but only the dependent variable
    :param exploratory_model_analysis: Boolean value to indicate whether to run the pycaret model compare method
    :return: 0: does not return anything as the trained model is stored locally
    """
    # First let's do some exploratory training analysis, to find out what type of models would perform well on the
    # dataset at hand. As the value to predict is a binary choice between True or False, getting hired or not, this
    # is a binary classification problem.

    if exploratory_model_analysis:
        # This piece of code will run a couple of models with PyCaret, and will generate summary statistics for the
        # models like accuracy, and F-Score.
        exp = ClassificationExperiment()
        x_data["decision"] = y_data
        exp.setup(x_data, target="decision", system_log="./data/logs.log")
        exp.compare_models()
        # From this exploratory training analysis comes forward that boosting methods are the best w.r.t. Accuracy, so
        # moving forward in the scikit learn pipeline the lightGBM Classifier will be used.

    classifier = Pipeline(steps=[("classifier", lgb.LGBMClassifier())])
    classifier.fit(x_data, y_data)

    ## Cross validation
    # potential improvement to use a cross validation instead of fit (to overcome overfitting) like
    from sklearn.model_selection import RandomizedSearchCV

    # param_dist = {
    #     'classifier__bagging_fraction': (0.5, 0.8),
    #     'classifier__feature_fraction': (0.5, 0.8),
    #     'classifier__max_depth': (10, 13),
    #     'classifier__min_data_in_leaf': (90, 120),
    #     'classifier__num_leaves': (1200, 1550)
    # }
    # search = RandomizedSearchCV(classifier, scoring='average_precision', cv=5,
    #                             n_iter=10, verbose=True, param_distributions=param_dist)
    # search.fit(x_data, y_data)
    # classifier = search.best_estimator_
    # store the preprocessing pipeline together with the classifier pipeline for later serving of the model
    complete_pipeline = Pipeline(steps=[("Preprocessing", preprocessing_pipeline), ("classifier", classifier)])

    ## Mitigation with fairlearn for gender bias
    # exponentiated_gradient = ExponentiatedGradient(
    #     estimator=complete_pipeline,
    #     constraints=DemographicParity(),
    #     sample_weight_name="classifier__classifier__sample_weight",
    # )
    # exponentiated_gradient.fit(raw_data, y_data, sensitive_features=raw_data["gender"])
    # complete_pipeline = exponentiated_gradient

    joblib.dump(complete_pipeline, "./data/model/recruitment_lightgbm_model.pkl")
    return 0


def evaluating_model(data: pd.DataFrame) -> int:
    """
    This function will evaluate a model whether it adheres to specific requirements we set. Specifically whether it
    adheres to bias requirements. If it doesn't adhere we need to go back in the previous steps and fix the
    preprocessing steps or the model hyperparameters. As this is an exmaple project, only suggestions will be done and
    not implemented as this would worsen the flow of the script.
    :param data: The data to look at the bias metrics to
    :return: 0
    """
    y_pred = serving_a_model(data)
    y_true = data.loc[:, "decision"]
    gender = data.loc[:, "gender"]
    dp = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=gender)
    sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=gender)
    # print(dp)
    # print(sr.by_group)
    # The difference in demographic parity is 0.25, this means that there is a 25% difference between the amount of
    # times that the 'lowest' gender gets selected compared to the highest. In this case that is between 'female' and
    # 'other'. This gives us reason to mitigate this bias for gender in the original model.
    return 0


def serving_a_model(data: pd.DataFrame) -> list:
    """
    This function will 'serve' the model, normally when serving a model one would include the preprocessing steps also
    within the model/pipeline. Therefore, this way of passing just the classifier and the preprocessed data would not
    suffice. Also, the serving of a model is generally that to an API/package you pass the input data and get the
    predicted result back.
    :param data: Pandas Dataframe containing the data
    :return: prediction_results: a list of boolean values for each datapoint a prediction
    """
    complete_pipeline = joblib.load("./data/model/recruitment_lightgbm_model.pkl")
    return complete_pipeline.predict(data)


def monitor_the_model(data):
    """
    If data drift occurs in the model we will see in the accuracy of the model (if we also have the true labels)
    declining. But even without the true labels we can also investigate whether the data distribution changes over
    time. More information on model monitoring [here](https://www.evidentlyai.com/ml-in-production/model-monitoring).
    :return: Testsuite object from evidently from which data drift on a specific column is investigated.
    """
    data_drift_column_tests = TestSuite(tests=[TestColumnDrift(column_name="gender", stattest="psi")])

    data_drift_column_tests.run(reference_data=data[:100], current_data=data[100:])
    return data_drift_column_tests.json()


def main() -> int:
    data = exploratory_data_analysis()
    train_data = data.sample(frac=0.8)
    evaluate_data = data.drop(train_data.index)
    preprocessor_pipeline, preprocessed_data = preprocessing_data(data=train_data)
    training_model(
        preprocessing_pipeline=preprocessor_pipeline,
        x_data=preprocessed_data,
        y_data=train_data["decision"],
        raw_data=train_data,
        exploratory_model_analysis=False,
    )
    evaluating_model(data=evaluate_data)
    serving_a_model(data=evaluate_data[0:10])
    monitor_the_model(data=evaluate_data)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
