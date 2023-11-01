import csv


def read_csv(filelocation) -> list[dict]:
    with open(filelocation) as f:
        return [row for row in csv.DictReader(f)]
