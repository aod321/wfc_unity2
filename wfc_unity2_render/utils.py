import json
import copy
import numpy as np
from map2graph import get_connectity_dict, tiles2data


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
