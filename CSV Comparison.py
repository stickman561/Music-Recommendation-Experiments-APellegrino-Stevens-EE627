import pandas as pa


def compare_csv(first_file: str, second_file: str):
    df_first = pa.read_csv(first_file, header=0, on_bad_lines="skip", )
    df_second = pa.read_csv(second_file, header=0, on_bad_lines="skip")

    # Access all lines where first contains same with second csv
    result_same = df_first[df_first.apply(tuple, 1).isin(df_second.apply(tuple,1))]
    print("Same")
    print(result_same)
    
    # Use Tilde to access all lines where first doesn't contain second csv
    result_diff = df_first[~df_first.apply(tuple, 1).isin(df_second.apply(tuple,1))]
    print("Difference")
    print(result_diff)


if __name__ == '__main__':
    compare_csv("..\\Homework\\Assignment 9\\Naive Bayes Predictions.csv", "..\\Homework\\Assignment 9\\Logistic Regression Predictions.csv")