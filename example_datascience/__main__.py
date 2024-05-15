import sys
import pandas as pd
import numpy as np
from pycaret.classification import ClassificationExperiment, create_model, check_fairness
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    count,
    demographic_parity_ratio,
    demographic_parity_difference,
)
from sklearn.metrics import accuracy_score, precision_score
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from aif360.datasets import StandardDataset
from data.download_dataset import download_from_kaggle
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
import joblib

np.random.seed(69)


def exploratory_data_analysis():
    """
    This function will do the exploratory data analysis (EDA) on the recruitment dataset of the University of Utrecht.
    In most project this is done in a Jupyter Notebook as this makes interaction easier, but a slice of the EDA is still
    implementated to give you an idea.
    :return: pd.Dataframe containing the data
    """
    datapath = "./data/recruitmentdataset-2022-1.3.csv"
    download_from_kaggle(datapath)

    df = pd.read_csv(datapath, index_col="Id")

    # print(df.head()) # The data is about whether based on some characteristic people are hired or not
    # print(df.shape) # There are 4000 datapoints with 13 feature columns and 1 label column
    # print(df[df.duplicated()]) # There are 39 duplicate values, but this is to be expected based on the categorical features
    # print(df.isna().sum()) # There are no missing values
    # print(df.describe()) # There are just 3 numerical features, which are the age, university grade, and languages.
    # print(df.columns)
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

    for col in cat_cols:
        categories = df.groupby(col).size()
        # print(categories)
    # From the categorical features we see that the non-binary gender is underrepresented although it is from the true
    # distribution of society at this moment. The same holds for nationality where Dutch is overrepresented. almost 3/4
    # of the applicants got rejected which we need to take into account in our model as well.
    return df


def preprocessing_data(data):
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


def training_model(preprocessing_pipeline, x_data, y_data, exploratory_model_analysis=False):
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
    joblib.dump(complete_pipeline, "./data/model/recruitment_lightgbm_model.pkl")
    return 0


def evaluating_model(classifier, x_data, y_data):
    """
    This function will evaluate a model whether it adheres to specific requirements we set. Specifically whether it
    adheres to bias requirements. If it doesn't adhere we need to go back in the previous steps and fix the
    preprocessing steps or the model hyperparameters. As this is an exmaple project, only suggestions will be done and
    not implemented as this would worsen the flow of the script.
    :param classifier: sklearn pipeline with the classifier model in it
    :param x_data:
    :param y_data:
    :return:
    """
    return 0


def serving_a_model(data):
    """
    This function will 'serve' the model, normally when serving a model one would include the preprocessing steps also
    within the model/pipeline. Therefore, this way of passing just the classifier and the preprocessed data would not
    suffice. Also the serving of a model is generally that to an API/package you pass the input data and get the
    predicted result back.
    :param data: Pandas Dataframe containing the data
    :return: prediction_results: a list of boolean values for each datapoint a prediction
    """
    complete_pipeline = joblib.load("./data/model/recruitment_lightgbm_model.pkl")
    return complete_pipeline.predict(data)


def monitor_the_model():
    """
    If data drift occurs in the model we will see in the accuracy of the model (if we also have the true labels)
    declining. But even without the true labels we can also investigate whether the data distribution changes over
    time.
    :return:
    """
    return 0


def convert_to_standard_dataset_for_aif360(df, target_label_name, scores_name=""):
    protected_attributes = []

    # columns from the dataset that we want to select for this Bias study
    selected_features = ["gender", "age"]

    privileged_classes = [[]]

    favorable_target_label = [1]

    # List of column names in the DataFrame which are to be expanded into one-hot vectors.
    categorical_features = ["gender", "nationality", "sport", "ind-degree", "company"]

    # create the `StandardDataset` object
    standard_dataset = StandardDataset(
        df=df,
        label_name=target_label_name,
        favorable_classes=favorable_target_label,
        scores_name=scores_name,
        protected_attribute_names=protected_attributes,
        privileged_classes=privileged_classes,
        categorical_features=categorical_features,
        features_to_keep=selected_features,
    )
    if scores_name == "":
        standard_dataset.scores = standard_dataset.labels.copy()

    return standard_dataset


def main() -> int:
    data = exploratory_data_analysis()
    preprocessor_pipeline, preprocessed_data = preprocessing_data(data)
    training_model(preprocessor_pipeline, preprocessed_data, data["decision"], exploratory_model_analysis=False)
    serving_a_model(data)

    # # models = exp.compare_models(include=['lr', 'dt', 'knn', 'catboost'])
    # models = exp.compare_models()
    #
    # # dt = exp.create_model('dt')
    # # catboost = exp.create_model('catboost')
    #
    # # 3. Evaluating models
    # # Experiment Tracking met mlflow (potentieel ook dataset tracking, maar voor nu doen we dat nog niet?)
    #
    # # save model
    # # exp.save_model(catboost, 'catboost_pipeline')
    # catboost = exp.load_model('catboost_pipeline')
    #
    # # 3.1 evaluating fairness-related metrics
    # # y_pred = exp.predict_model(catboost, data=unseen_data).loc[:, "prediction_label"]
    # y_pred = exp.predict_model(catboost, data=unseen_data)
    # y_true = unseen_data.loc[:, "decision"]
    # gender = unseen_data.loc[:, "gender"]
    # # dp = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=gender)
    # # print(dp)
    # # dp = demographic_parity_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=gender)
    # # Deze geeft een ratio van 0.67 aan wat betekent dat er vaker males worden geselecteerd dan females door het model
    # # mfx = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=gender)
    # # sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=gender)
    # # print(dp)
    # # print(sr.by_group)
    #
    # # ## check fairness out of the box of pycaret
    # # catboost_fairness = exp.check_fairness(catboost, sensitive_features=['gender'])
    # # print(catboost_fairness)
    #
    # ## check mitigation of demographic disparity of gender via fairlearn
    # # constraint = DemographicParity()
    # # mitigator = ExponentiatedGradient(catboost, constraint)
    # # exp.traiN_model
    # X_train_transformed = exp.get_config(variable="X_train_transformed")
    # train_gender_transformed = exp.get_config(variable="X_train_transformed").loc[:, ["gender_female", "gender_male", "gender_other"]]
    # y_train_transformed = exp.get_config(variable="y_train_transformed")
    # # mitigator.fit(X_train_transformed, y_train_transformed, sensitive_features=train_gender_transformed)
    # # y_pred_mitigated = mitigator.predict(unseen_data)
    # # sr_mitigated = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=test_gender)
    # # print(sr_mitigated.overall)
    # # print(sr_mitigated.by_group)
    # #
    #
    #
    # ## plotting of the fairness metrics w.r.t. gender
    # # metrics = {
    # #     "accuracy": accuracy_score,
    # #     "precision": precision_score,
    # #     "false positive rate": false_positive_rate,
    # #     "false negative rate": false_negative_rate,
    # #     "selection rate": selection_rate,
    # #     "count": count,
    # # }
    # # metric_frame = MetricFrame(
    # #     metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=gender
    # # )
    # # fig = metric_frame.by_group.plot.bar(
    # #     subplots=True,
    # #     layout=[3, 3],
    # #     legend=False,
    # #     figsize=[12, 8],
    # #     title="Show all metrics",
    # # )
    # # fig[0][0].figure.savefig("bias.png")
    # # print(unseen_data.loc[unseen_data['gender'] == 'other'])
    #
    # # 4. Analysis & Interpretability
    # # exp.plot_model(catboost, plot='confusion_matrix')
    # # exp.plot_model(catboost, plot='auc')
    # exp.plot_model(catboost, plot='feature', save=True)
    #
    # # 4. Serving a result
    #
    # # 5. Monitoring
    # pred_dataset = exp.predict_model(catboost, data=unseen_data)
    # X_train_transformed = exp.get_config(variable="X_train_transformed")
    # print(X_train_transformed.columns)
    # print(pred_dataset.columns)
    ### TESTING WITH AIF360 packages

    # Metric for the original dataset
    # standard_dataset_pred_aif360 = convert_to_standard_dataset_for_aif360(exp.predict_model(catboost, data=unseen_data),
    #                                                        target_label_name='prediction_label',
    #                                                        scores_name='prediction_score')
    # # metric_orig_train = BinaryLabelDatasetMetric(X_train_transformed,
    # #                                              unprivileged_groups=[{'Gender': 'other'}],
    # #                                              privileged_groups=[{'Gender': 'male'}])
    # print(metric_orig_train)

    # bias mitigating results
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
