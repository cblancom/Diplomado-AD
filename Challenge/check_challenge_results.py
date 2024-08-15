import os
import pandas as pd
from sklearn.metrics import f1_score


class CheckChallengeResults:
    """
    Class for evaluating multiple predictions against a test dataset.

    This class iterates through different subfolders in a specified directory and searches for two files: 'y_pred1.csv' and 'y_pred2.csv' within each subfolder. These files contain predictions based on a dataset. The class then calculates the F1 score for each prediction and stores the results in a dictionary.
    """

    def __init__(self, groups_path, true_results_path):
        """
        Initialization of the variables to be used. The necessary functions are called in the appropriate order.

        :param groups_path str: Directory where the folders will be searched.
        :param true_results_path str: Path where the actual results are found.
        """
        self.groups_path = groups_path
        self.true_results_path = true_results_path

        self.get_files()
        self.get_F1_scores()

    def get_files(
        self,
    ):
        """
        This function iterates through the folders and determines which ones contain both valid files, which ones have only one file, and which ones contain none.
        """
        self.csv_dict = {}
        for folder_name in os.listdir(self.groups_path):
            folder_path = os.path.join(self.groups_path, folder_name)

            if os.path.isdir(folder_path):
                # Initialize a list to store the CSV data
                csv_files = {}
                for csv_file in ["y_pred1.csv", "y_pred2.csv"]:
                    file_path = os.path.join(folder_path, csv_file)
                    if os.path.isfile(file_path):
                        # Read the CSV file and store it in the dictionary
                        csv_files[csv_file] = pd.read_csv(file_path)
                    self.csv_dict[folder_name] = csv_files

    def get_F1_scores(
        self,
    ):
        """
        This function iterates through the files with the correct names (determined in the previous method), calls another method to compute the F1 scores, and sorts the results.
        """
        self.results_dict = {}
        # Load and reorder y_test.csv with respect to the 'code' column
        y_test = pd.read_csv(self.true_results_path)

        for group_results_names in list(self.csv_dict.keys()):
            f1_scores = []
            group_csv = self.csv_dict[group_results_names]
            group_csv_names = list(group_csv.keys())

            if len(group_csv_names) == 0:
                f1_scores = [
                    "Sin archivos válidos",
                    "Sin archivos válidos",
                ]

            elif len(group_csv_names) == 1:
                y_pred = group_csv[group_csv_names[0]]
                f1_score_result = self.compute_f1(y_test, y_pred)
                f1_scores = [f1_score_result, "Falta un archivo"]

            elif len(group_csv_names) == 2:
                for csv_file in group_csv_names:
                    y_pred = group_csv[csv_file]
                    f1_scores.append(self.compute_f1(y_test, y_pred))

            self.results_dict[group_results_names] = f1_scores

    def compute_f1(self, y_true, y_pred):
        """
        Additional method used to calculate the F1 score.

        At this stage, we validate that both the predictions and the actual data have the same IDs. If the IDs match, we calculate the F1 score and return the value to the previous method. If the IDs do not match, an appropriate message is returned.

        :param y_true pd.DataFrame: Dataframe with true data
        :param y_pred pd.DataFrame: Dataframe with predicted data
        """
        label_mapping = {"si": 1, "no": 0}
        try:
            y_true = y_true.sort_values(by="codigo_cliente").reset_index(drop=True)
            y_pred = y_pred.sort_values(by="codigo_cliente").reset_index(drop=True)
            if y_true["codigo_cliente"].equals(y_pred["codigo_cliente"]):
                y_true_map = y_true["resultado"].map(label_mapping)
                y_pred_map = y_pred["resultado"].map(label_mapping)
                f1_score_result = f1_score(y_true_map, y_pred_map)
            else:
                f1_score_result = "Los clientes son diferentes"

        except Exception as error:
            f1_score_result = str(error)
        return f1_score_result
