from map2graph import map2digraph, tiles2data, get_map_shortest_length_dist, get_all_pair_shortest_path
from glob import glob
import networkx as nx
import numpy as np
from utils import calc_entropy,  length_list_to_dist, cs_divergence
import fastwfc

# 求空地分布
wfc = fastwfc.XLandWFC("samples.xml")
empty = wfc.get_ids_from_wave(wfc.build_a_open_area_wave())
empty_dist = get_map_shortest_length_dist(map2digraph(tiles2data(empty)), norm=False)

# 生成一张地图,并求其分布
ids,_ = wfc.generate(out_img=False)
base_wave = wfc.wave_from_id(ids)
dist = get_map_shortest_length_dist(map2digraph(tiles2data(ids)), norm=False)

# 求两个分布的cs散度
divergence = cs_divergence(empty_dist, dist)
print(f"empty divergence: {divergence}")
