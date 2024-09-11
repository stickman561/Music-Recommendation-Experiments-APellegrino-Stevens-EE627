import pandas as pd

main_predictions = input("Enter name of main prediction file: ")
fallback_predictions = input("Enter name of fallback prediction file: ")

def merge_csv_files(file1_path, file2_path, output_path):
    # Read CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge DataFrames based on the 'TrackID' column
    merged_df = pd.merge(df1, df2, on='TrackID', how='outer', suffixes=('_file1', '_file2'))

    # Fill NaN values in the 'Predictor_file1' column with values from 'Predictor_file2'
    merged_df['Predictor_file1'].fillna(merged_df['Predictor_file2'], inplace=True)

    # Drop the 'Predictor_file2' column
    merged_df.drop(columns=['Predictor_file2'], inplace=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)

# Example usage:
output_path = 'merged_output.csv'

merge_csv_files(main_predictions, fallback_predictions, output_path)