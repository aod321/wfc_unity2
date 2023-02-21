import argparse
import os
from map2graph import map2digraph, tiles2data, get_map_shortest_length_dist, get_all_pair_shortest_path
from glob import glob
import networkx as nx
import numpy as np
from utils import calc_entropy, json_to_tileid, tileid_to_json
from natsort import natsorted
import fastwfc


def DiGraphStats(DiGraph):
    all_pair_shortest_path, all_pair_shortest_path_dict = get_all_pair_shortest_path(DiGraph, return_dict=True)
    all_pair_length = []
    for pair in all_pair_shortest_path:
        all_pair_length.append(len(pair))
    print("mean shortest path length: ", np.mean(all_pair_length))
    entropy_of_shortest_path_dist = calc_entropy(get_map_shortest_length_dist(DiGraph, norm=False).reshape(-1))
    print("entropy of shortest path dist: ", entropy_of_shortest_path_dist)
    all_pair_shortest_path, all_pair_shortest_path_dict = get_all_pair_shortest_path(DiGraph, return_dict=True)
    no_path_pair_count = 0
    for i in DiGraph.nodes:
        for j in DiGraph.nodes:
            if i!=j and j not in all_pair_shortest_path_dict[i].keys():
                no_path_pair_count += 1
    # print(all_pair_shortest_path_dict[0][18])
    print("no path pair count: ", no_path_pair_count)
    no_path_pair_rate = no_path_pair_count /(len(DiGraph.nodes)**2)
    print(f"no path pair rate: {no_path_pair_rate}")
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_json", help="put json file here")
    parser.add_argument("--map_json_folder", help="put a folder containing json file here")
    parser.add_argument("--render_in_unity", action="store_true", help="render in unity")
    parser.add_argument("--generate", action="store_true",help="generate")

    args = parser.parse_args()
    print(args)
    generate = args.generate
    render_in_unity = args.render_in_unity
    unity3d_env = None
    wfc = fastwfc.XLandWFC("samples.xml")
    if render_in_unity:
        from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
        file_name = "./windows_build/1209_windows_build/tilemap_render.exe"
        unity3d_env = WFCUnity3DEnv(file_name=file_name)
        unity3d_env.render_in_unity()
    if args.map_json:
        state = 1
    elif args.map_json_folder:
        state = 2
    elif generate:
        state = 3
    else:
        raise Exception("Please specify either --map_json or --map_json_folder or --generate")

    if state == 1:
        map_json = args.map_json
        tiles = json_to_tileid(map_json)
        DiGraph = map2digraph(tiles2data(tiles))
        print(DiGraph)
        DiGraphStats(DiGraph=DiGraph)

    elif state == 2:
        json_path = os.path.join(args.map_json_folder, "*.json")
        all_files = natsorted(glob(json_path))
        # print(all_files)
        for map_json in all_files:
            print(map_json)
            map_in = json_to_tileid(map_json)
            tiles = json_to_tileid(map_json)
            wave = wfc.wave_from_id(tiles)
            DiGraph = map2digraph(tiles2data(tiles))
            print(DiGraph)
            DiGraphStats(DiGraph=DiGraph)
            print(render_in_unity)
            if render_in_unity:
                unity3d_env.set_wave(wave=wave)
                unity3d_env.render_in_unity()
        
    elif state == 3:
        ids, _ = fastwfc.generate(out_img=False)
        wave = wfc.wave_from_id(ids)
        DiGraph = map2digraph(tiles2data(ids))
        print(DiGraph)
        DiGraphStats(DiGraph=DiGraph)
        

    if state != 2:
        if render_in_unity:
            unity3d_env.set_wave(wave=wave)
            unity3d_env.render_in_unity()
