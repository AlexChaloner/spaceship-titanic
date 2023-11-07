import numpy as np
import pandas as pd


def get_data():
    TRAIN_DATA_FILE = "data/train.csv"

    train_data_df = pd.read_csv(TRAIN_DATA_FILE)

    train_data_df = process_data(train_data_df)

    return train_data_df


def get_train_data(filter_columns=True):
    df = get_data()
    # Based on human judgment, the important columns
    important_columns = [
        "PassengerIdSplit1",
        "CryoSleep",
        "VIP",
        "TotalSpend",
        "Cabin1",
        "Cabin2",
        "Cabin3",
        "Age",
        "Destination",
        "HomePlanet",
        "Transported"
    ]

    if filter_columns:
        df = df[important_columns]
    # categorical_columns = df.select_dtypes(include=['object']).columns

    # # Apply one-hot encoding to the categorical columns
    # df = pd.get_dummies(df, columns=categorical_columns)

    return df


def process_data(df):
    # Condense all spending into one, under assumption each one does not give enough detail
    spending_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spending_columns].sum(axis=1)

    # Process Cabins
    df[['Cabin1', 'Cabin2', 'Cabin3']] = df['Cabin'].str.split('/', n=3, expand=True)
    df["Cabin2"] = df["Cabin2"].fillna(-1).astype(int)

    # Remove the 'full_name' column under assumption it is noisy
    df = df.drop('Cabin', axis=1)

    # Harvest the first half the of the passenger ID, as it has data
    df[['PassengerIdSplit1', 'PassengerIdSplit2']] = df['PassengerId'].str.split('_', n=1, expand=True)
    df['PassengerIdSplit1'] = df['PassengerIdSplit1'].fillna(-1).astype(int)

    return df
