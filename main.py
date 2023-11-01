from utils import read_csv
from data import TitanicData
import matplotlib.pyplot as plt


def main():
    csv_object = read_csv("data/train.csv")
    train_data = TitanicData(csv_object).data

    # Let's visualise the training data
    total = len(train_data)
    num_transported = len([row for row in train_data if row.transported])

    print(num_transported, total)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.scatter([row.age for row in train_data], [row.transported for row in train_data])
    # plt.show()

    # plt.scatter(
    #     [row.cabin[2] for row in train_data], [row.transported for row in train_data],
    #     alpha=0.005)
    # plt.show()

    # plt.scatter(
    #     [int(row.vip) for row in train_data], [row.transported for row in train_data],
    #     alpha=0.005)
    # plt.show()

    # Define colors for transported and Non-transported
    colors = {False: 'red', True: 'blue'}

    # Create a scatter plot with color-coding based on VIP status
    plt.scatter([row.age for row in train_data],
                [1 if row.cryo_sleep else 0 for row in train_data],
                c=[colors[row.transported] for row in train_data],
                label=[row.transported for row in train_data],
                alpha=0.01)
    plt.show()



if __name__ == "__main__":
    main()