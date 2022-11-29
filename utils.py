# date: 2022-11-24
# author: Yin Zi
# Some helper functions

import numpy as np
import networkx as nx
from math import sqrt
from math import log
import json
import copy
from map2graph import get_connectity_dict


# modified, original code from https://gist.github.com/Jarino/cb6d9b39abcf773a1fb0e9a90ee67db9
def cs_divergence(p1, p2):
    """
    Calculates the Cauchy-Schwarz divergence between two probabilities distribution. CS divergence is symmetrical,
    hence the order of the arguments does not matter. The result is from interval [0, infinity], 
    where 0 is obtained when the two probabilities distributions are same.
    Args:
        p1 (numpy array): first pdfs
        p2 (numpy array): second pdfs
    Returns:
        float: CS divergence
    """
    p1_computed = p1
    p2_computed = p2
    numerator = sum(p1_computed * p2_computed)
    denominator = sqrt(sum(p1_computed ** 2) * sum(p2_computed**2))
    return -log(numerator/denominator)

# calculate a entropy of a distribution
def calc_entropy(dist):
    entropy = 0
    for i in range(len(dist)):
        if dist[i] == 0:
            continue
        entropy += dist[i] * np.log(dist[i])
    return -entropy

def length_list_to_dist(length_list: np.ndarray, min_length=81) -> np.ndarray:
    return np.bincount(length_list.reshape(-1), minlength=min_length) / np.bincount(length_list.reshape(-1), minlength=min_length).sum()

def tileid_to_json(tile_id: list, save_path: str):
    json_data ={}
    wave_list = []
    for i in range(81):
        wave_list.append(tile_id[i][0])
        wave_list.append(tile_id[i][1])
    json_data["data"] = wave_list
    json.dump(json_data, open(save_path, 'w'))
    return json_data

def json_to_tileid(filename):
    with open(filename) as json_file:
        json_file = json.load(json_file)
        data =copy.deepcopy(json_file['data'])
    data1 = []
    for i in range(len(data)//2):
        data1.append([data[2*i], data[2*i+1]])
    return data1

def tile2heightdist(tile_ids, return_height_map=False):
    _,all_height_dict = get_connectity_dict(out_height_dict=True)
    id_to_height = {}
    for tile in tile_ids:
        current_tile_index = tile[0]
        for height,ids in all_height_dict.items():
            if current_tile_index in ids:
                id_to_height[current_tile_index] = height
                break
    all_height_map = []
    for tile in tile_ids:
        tile_id = tile[0]
        tile_height = id_to_height[tile_id]
        all_height_map.append(tile_height)
    np_all_height_map = np.array(all_height_map)
    if return_height_map:
        return np_all_height_map
    else:
        return np.bincount(np_all_height_map, minlength=7)

"""
    | ID          | Name                 | Height |
    | ----------- | -------------------- | ------ |
    | 0,1,2,3     | gray_cube            | 1      |
    | 4,5,6,7     | blue_cube            | 2      |
    | 8,9,10,11   | yellow_cube          | 3      |
    | 12,13,14,15 | orange_cube          | 4      |
    | 16,17,18,19 | red_cube             | 5      |
    | 20,21,22,23 | white_cube           | 6      |
    | 24,25,26,27 | corner_blue_gray     | 1.5      |
    | 28,29,30,31 | corner_yellow_blue   | 2.5      |
    | 32,33,34,35 | corner_orange_yellow | 3.5      |
    | 36,37,38,39 | corner_red_orange    | 4.5      |
    | 40,41,42,43 | corner_white_red     | 5.5      |
    | 44,45,46,47 | corner_yellow_gray   | 2      |
    | 48,49,50,51 | corner_orange_blue   | 3      |
    | 52,53,54,55 | corner_red_yellow    | 4      |
    | 56,57,58,59 | corner_white_orange  | 5      |
    | 60,61,62,63 | corner_orange_gray   | 2.5      |
    | 64,65,66,67 | corner_red_blue      | 3.5      |
    | 68,69,70,71 | corner_white_yellow  | 4.5      |
    | 72,73,74,75 | corner_red_gray      | 3      |
    | 76,77,78,79 | corner_white_blue    | 4      |
    | 80,81,82,83 | corner_white_gray    | 3.5      |
    | 84,85,86,87 | ramp_blue            | 2      |
    | 88,89,90,91 | ramp_yellow          | 3      |
    | 92,93,94,95 | ramp_orange          | 4      |
    | 96,97,98,99 | ramp_red             | 5      |
    | 100,101,102,103 | ramp_white          | 6      |
"""
def tilemap2heightmap(map_ids):
    # height_dict = {
    #     0: set(),
    #     1: set([0,1,2,3]),
    #     1.5: set([24, 25, 26, 27]),
    #     2: set([4, 5, 6, 7, 44, 45, 46, 47, 84, 85, 86, 87]),
    #     2.5: set([28, 29, 30, 31, 60, 61, 62, 63]),
    #     3: set([8, 9, 10, 11, 48, 49, 50, 51, 72, 73, 74, 75, 88, 89, 90, 91]),
    #     3.5: set([32, 33, 34, 35, 64, 65, 66, 67, 80, 81, 82, 83]),
    #     4: set([12, 13, 14, 15, 52, 53, 54, 55, 76, 77, 78, 79, 92, 93, 94, 95]),
    #     4.5: set([36, 37, 38, 39, 68, 69, 70, 71]),
    #     5: set([16, 17, 18, 19, 56, 57, 58, 59, 96, 97, 98, 99]),
    #     5.5: set([40, 41, 42, 43]),
    #     6: set([20, 21, 22, 23, 100, 101, 102, 103]),
    #     7: set()
    # }
    _, height_dict= get_connectity_dict(out_height_dict=True)
    id_to_height = {}
    for height, ids in height_dict.items():
        for id in ids:
            id_to_height[id] = height
    all_height_map = []
    for tile in map_ids:
        tile_id = tile[0]
        tile_height = id_to_height[tile_id]
        all_height_map.append(tile_height)
    np_all_height_map = np.array(all_height_map)
    return np_all_height_map

def id_height_distance(id1, id2):
    height_map1 = tilemap2heightmap(id1) / 6
    height_map2 = tilemap2heightmap(id2) / 6
    return np.linalg.norm(height_map2 - height_map1)
