import kaggle
import os.path


def download_from_kaggle(datapath):
    """
    Downloads the dataset from Kaggle, the authentication needs to be stored locally in the home directy as a
    kaggle.json more info on the website of Kaggle
    [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).
    :param datapath: The path to the datafile
    :return: 0
    """
    if not os.path.isfile(datapath):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('ictinstitute/utrecht-fairness-recruitment-dataset',
                                          path='./',
                                          unzip=True)
    return 0
