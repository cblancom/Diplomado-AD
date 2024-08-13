import os
import pandas as pd
from sklearn.metrics import f1_score


class CheckChallengeResults:
    def __init__(self, groups_path, true_results_path):
        self.groups_path = groups_path
        self.true_results_path = true_results_path

        self.get_files()
        self.get_F1_scores()

    def get_files(
        self,
    ):
        self.csv_dict = {}
        for folder_name in os.listdir(self.groups_path):
            folder_path = os.path.join(self.groups_path, folder_name)
            # print(folder_path)

            if os.path.isdir(folder_path):
                # Initialize a list to store the CSV data
                csv_files = {}
                for csv_file in ["y_pred1.csv", "y_pred2.csv"]:
                    file_path = os.path.join(folder_path, csv_file)
                    if os.path.isfile(file_path):
                        # Read the CSV file and store it in the dictionary
                        csv_files[csv_file] = pd.read_csv(file_path)
                    # if csv_files:
                    self.csv_dict[folder_name] = csv_files

    def get_F1_scores(
        self,
    ):
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
        label_mapping = {"si": 1, "no": 0}
        y_true = y_true.sort_values(by="codigo_cliente").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="codigo_cliente").reset_index(drop=True)

        if y_true["codigo_cliente"].equals(y_pred["codigo_cliente"]):
            try:
                y_true_map = y_true["resultado"].map(label_mapping)
                y_pred_map = y_pred["resultado"].map(label_mapping)
                f1_score_result = f1_score(y_true_map, y_pred_map)
            except:
                f1_score_result = "El archivo no tiene el formato adecuado"
        else:
            f1_score_result = "El archivo no tiene el formato adecuado"
        return f1_score_result
