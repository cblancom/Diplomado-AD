from check_challenge_results import CheckChallengeResults
import pandas as pd

if __name__ == "__main__":
    groups_path = "./Groups/"
    true_labels = "./y_true.csv"
    CCR = CheckChallengeResults(groups_path=groups_path, true_results_path=true_labels)
    print(pd.DataFrame(CCR.results_dict))
