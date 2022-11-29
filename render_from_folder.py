from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
import fastwfc
import json
from utils import json_to_tileid, tileid_to_json
import os
from glob import glob
from natsort import natsorted
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--map_json_folder", help="put a folder containing json file here")
args = parser.parse_args()
print(args)

unity3denv = WFCUnity3DEnv()
wfc = fastwfc.XLandWFC("samples.xml")
all_json_file_list = natsorted(glob(os.path.join(args.map_json_folder, "*.json")))
for json_file in all_json_file_list:
    print(json_file)
    tiles = json_to_tileid(json_file)
    wave = wfc.wave_from_id(tiles)
    unity3denv.set_wave(wave=wave)
    unity3denv.render_in_unity()
    time.sleep(0.5)




