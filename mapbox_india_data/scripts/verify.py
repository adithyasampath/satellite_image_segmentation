"""
18 August, 2019
Pranav Bijapur

Simple script to verify that file names in two directories (tiles and labels) match.
"""

import os

PATH_TILES = "mapbox_india_data/handlabelled_data/original/tiles"
PATH_LABELS = "mapbox_india_data/handlabelled_data/original/labels"


def get_list_of_files(dir_name):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(dir_name):
        list_of_files += [os.path.join(dirpath, file) for file in filenames]
    return list_of_files


def get_unmatched_files(tiles, labels):
    union = set(tiles + labels)
    intersection = set(tiles) & set(labels)
    print("Unmatched files are:", union - intersection)


tiles = get_list_of_files(PATH_TILES)
labels = get_list_of_files(PATH_LABELS)

tiles = [tile.split("/")[-1] for tile in tiles]
labels = [label.split("/")[-1] for label in labels]
tiles.sort()
labels.sort()

if len(tiles) != len(labels):
    print("Number of tiles are {} and number of labels are {}".format(len(tiles), len(labels)))
    get_unmatched_files(tiles, labels)
    print("Exiting...")
    exit()

for tile, label in zip(tiles, labels):
    if tile.split("/")[-1] != label.split("/")[-1]:
        print("{} tile and {} label do not match".format(tile, label))
        print("Exiting...")
        exit()

print("All file names successfully verified.")
print("Total {} tiles and labels each.".format(len(tiles)))
