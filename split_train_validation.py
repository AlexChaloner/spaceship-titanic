import pandas as pd


def main():
    df = pd.read_csv("data/original_train.csv", index_col="PassengerId")
    val = df.sample(frac=0.2)
    train = df.drop(val.index)

    val.to_csv("data/validation.csv")
    train.to_csv("data/train.csv")


if __name__ == "__main__":
    main()
