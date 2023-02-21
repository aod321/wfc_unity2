# date: 2022-11-24
# author: Yin Zi
# Some helper functions

import numpy as np
import networkx as nx
from math import sqrt
from math import log
import json
import copy
from map2graph import get_connectity_dict, tiles2data


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
    for i in range(len(tile_id)):
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
    map_data= tiles2data(tile_ids)
    _,all_height_dict = get_connectity_dict(out_height_dict=True)
    id_to_height = {}
    for tile in map_data:
        for height,ids in all_height_dict.items():
            if tile in ids:
                id_to_height[tile] = height
                break
    all_height_map = []
    for tile in map_data:
        tile_height = id_to_height[tile]
        all_height_map.append(tile_height)
    np_all_height_map = np.array(all_height_map)
    if return_height_map:
        return np_all_height_map
    else:
        return np.bincount(np_all_height_map, minlength=7)

def tilemap2heightmap(map_ids):
    map_data = tiles2data(map_ids)
    _, height_dict= get_connectity_dict(out_height_dict=True)
    id_to_height = {}
    for height, ids in height_dict.items():
        for id in ids:
            id_to_height[id] = height
    all_height_map = []
    for tile in map_data:
        tile_height = id_to_height[tile]
        all_height_map.append(tile_height)
    np_all_height_map = np.array(all_height_map)
    return np_all_height_map

def id_height_distance(id1, id2):
    height_map1 = tilemap2heightmap(id1) / 6
    height_map2 = tilemap2heightmap(id2) / 6
    return np.linalg.norm(height_map2 - height_map1)
