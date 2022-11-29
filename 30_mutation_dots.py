import fastwfc
import matplotlib.pyplot as plt
import numpy as np
import random
from map2graph import tiles2data, get_connectity_dict, map2digraph, get_map_shortest_length_dist
from utils import id_height_distance, tilemap2heightmap, cs_divergence
from datetime import datetime

def generate_mutated_waves(old_wave, mutate_count):
    mutated_waves = []
    wave = old_wave
    for i in range(mutate_count):
        seed,img = wfc.mutate(base_wave=wave, out_img=False, iter_count=1, new_weight=162.0)
        wave = wfc.wave_from_id(seed)
    mutated_waves.append(wave) 
    return mutated_waves

def generate_all_mutes(dot_counts, mutate_counts):
    all_muts = {}
    init_waves = {count: [wfc.wave_from_id(wfc.generate(out_img=False)[0]) for _ in range(dot_counts[idx])] for idx,count in enumerate(mutate_counts)}
    for mutate_idx,mutate_count in enumerate(mutate_counts):
        all_muts[mutate_count] = [generate_mutated_waves(init_waves[mutate_count][i], mutate_count=mutate_count) for i in range(dot_counts[mutate_idx])]
    return all_muts, init_waves

def wave_height_distance(wave1, wave2):
    id1 = wfc.get_ids_from_wave(wave1)
    id2 = wfc.get_ids_from_wave(wave2)
    return id_height_distance(id1, id2)

def wave_height_map(wave):
    ids = wfc.get_ids_from_wave(wave)
    return tilemap2heightmap(ids) / 6.0
    

def all_muts_to_height_list(all_muts):
    all_w_list = {}
    for key in all_muts.keys():
        all_w_list[key] = []
        for muts in all_muts[key]:
            all_w_list[key].append(np.array(wave_height_map(muts[0])))
    return all_w_list

def all_mutes_to_digraphs(all_muts):
    all_digraphs = {}
    for key in all_muts.keys():
        all_digraphs[key] = []
        for muts in all_muts[key]:
            all_digraphs[key].append([map2digraph(tiles2data(wfc.get_ids_from_wave(muts[0]))) for wave in muts])
    return all_digraphs

def digraph_to_dist(digraph):
    return get_map_shortest_length_dist(digraph, norm=False)

def all_digraphs_to_dist(all_digraphs):
    all_dist = {}
    for key in all_digraphs.keys():
        all_dist[key] = []
        for digraphs in all_digraphs[key]:
            all_dist[key].append([digraph_to_dist(digraph) for digraph in digraphs])
    return all_dist

if __name__ == '__main__':
    wfc = fastwfc.XLandWFC("samples.xml")
    dot_counts = [50, 50, 50, 50]
    mutate_counts = [1, 10, 20, 30]
    dot_color = ['blue','#460087','#7c0062','red']
    all_muts, init_waves = generate_all_mutes(dot_counts, mutate_counts)

    # X轴待减项
    all_height_list = all_muts_to_height_list(all_muts)
    all_dg = all_mutes_to_digraphs(all_muts)
    all_dist = all_digraphs_to_dist(all_dg)

    # x, y 对应初始值
    init_height_list = {}
    init_dist_list = {}
    for key in init_waves.keys():
        init_height_list[key] = []
        init_dist_list[key] = []
        for wave in init_waves[key]:
            init_height_list[key].append(np.array(wave_height_map(wave)))
            init_dist_list[key].append(digraph_to_dist(map2digraph(tiles2data(wfc.get_ids_from_wave(wave)))))
    # # 分别求x, y 与初始值的对应距离
    all_x = {}
    all_y = {}
    for key in init_height_list.keys():
        all_x[key] = []
        for idx,w_list in enumerate(all_height_list[key]):
            init_w_array = np.array(init_height_list[key][idx])
            w_array = np.array(w_list)
            distance = np.linalg.norm(w_array - init_w_array)
            all_x[key].append(distance)
        all_y[key] = []
        for idx,dist_list in enumerate(all_dist[key]):
            init_dist = init_dist_list[key][idx]
            divergence = cs_divergence(np.array(dist_list[0]), np.array(init_dist))
            all_y[key].append(divergence)

    # 画图，变化次数分别为1, 10, 20, 30
    plt.xlim(0, 4.0)
    plt.ylim(0, 1.0)
    plt.scatter(all_x[1], all_y[1], c=dot_color[0], label='1')
    plt.scatter(all_x[10], all_y[10], c=dot_color[1], label='10')
    plt.scatter(all_x[20], all_y[20], c=dot_color[2], label='20')
    plt.scatter(all_x[30], all_y[30], c=dot_color[3], label='30')
    # show label
    plt.legend()
    plt.savefig(f"{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.pause(0.01)