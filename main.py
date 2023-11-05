from utils import read_csv
from data import TitanicData
import matplotlib.pyplot as plt
import pandas as pd
import math


def main():
    TRAIN_DATA_FILE = "data/train.csv"
    csv_object = read_csv(TRAIN_DATA_FILE)
    train_data = TitanicData(csv_object)

    train_data_df = pd.read_csv(TRAIN_DATA_FILE, index_col="PassengerId")

    train_data_df[['Cabin1', 'Cabin2', 'Cabin3']] = train_data_df['Cabin'].str.split('/', n=3, expand=True)
    train_data_df["Cabin2"] = train_data_df["Cabin2"].fillna(0).astype(int)
    # Remove the 'full_name' column
    train_data_df = train_data_df.drop('Cabin', axis=1)

    display_all(train_data_df)
    display_comparisons_discrete(train_data_df, "Transported")
    display_comparisons_discrete(train_data_df, "VIP")
    display_comparisons_discrete(train_data_df, "CryoSleep")



def display_all(df: pd.DataFrame):
    fig = plt.figure()
    add_charts_all_columns(df, fig)
    plt.show()


def display_comparisons_discrete(df: pd.DataFrame, comparison_column: str):
    unique_values = df[comparison_column].unique()
    print(unique_values)
    if len(unique_values) > 4:
        print("Warning: More than 4 unique values. Check if you really want to do this?")
        return
    for v in unique_values:
        fig = plt.figure()
        new_df = df[df[comparison_column] == v].copy(deep=True)
        new_df = new_df.drop(comparison_column, axis=1)
        add_charts_all_columns(new_df, fig)
        fig.suptitle(f"{comparison_column} = {v}")
    plt.show()


def add_charts_all_columns(df: pd.DataFrame, fig: plt.Figure):
    COLUMN_NUMBER = len(df.columns)
    intSqrtCol = math.ceil(math.sqrt(COLUMN_NUMBER))

    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(intSqrtCol, intSqrtCol, i+1)
        ax.set_title(column)

        if df[column].dtype in ['int32', 'int64', 'float64']:
            ax.hist(df[column], bins=20)
        elif column == "Name":
            continue
        else:
            counts = df[column].value_counts()
            counts = counts.sort_index()
            ax.bar(counts.index, counts)
    plt.subplots_adjust(wspace=1, hspace=1)
    

if __name__ == "__main__":
    main()
