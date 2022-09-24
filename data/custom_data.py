import os
import glob
import sys
sys.path.append(".")
import pandas as pd


def make_folder(path):
    if os.path.isdir(path):
        return
    os.mkdir(path)


def main(path_to_csv_file, image_dir = "data/train", new_dir = "data/custom_data"):
    data = pd.read_csv(path_to_csv_file)
    print(data.head(5))
    print(data.info())
    labels = data['label'].unique()
    print(f"There are {len(data)} in the training set!")
    print(f"Number of people: {len(labels)}")
    print(f"Number of people: {data['label'].value_counts()}")

    #########################################################

    make_folder(new_dir)
    for label in labels:
        label_folder = os.path.join(new_dir, f"{label}")
        make_folder(label_folder)
        all_file_names_in_this_label = data[data['label']==label]['image']
        for name in all_file_names_in_this_label:
            file = os.path.join(image_dir, name)
            new_file = os.path.join(label_folder, name)
            try:
                os.rename(file, new_file)
            except:
                continue

if __name__ == "__main__":
    main("data/train.csv")