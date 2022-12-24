import cv2
import os
import argparse
import fastwfc
from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
from utils import tileid_to_json, json_to_tileid, tilemap2heightmap
from datetime import datetime
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--os", type=str, required=True, default="macos", help="macos or windows")
arg_parser.add_argument("--base", type=str, required=True, help="base map")
args = arg_parser.parse_args()

if args.os == "windows":
    file_name = "./windows_build/1209_windows_build/tilemap_render.exe"
else:
    file_name = "./mac_build/1209_mac_built.app/Contents/MacOS/tilemap_render"

img_out_folder = "sameheight_out_images"
json_out_folder = "sameheight_out_jsons"
os.makedirs(img_out_folder, exist_ok=True)
os.makedirs(json_out_folder, exist_ok=True)
try:
    wfc = fastwfc.XLandWFC("samples.xml")
    map_data = json_to_tileid(args.base)
    height_dist = np.bincount(tilemap2heightmap(map_data), minlength=7)
    print(f"base height dist: {height_dist}")
    new_ids,new_img = wfc.generate(out_img=True) 
    new_heihgt_dist = np.bincount(tilemap2heightmap(new_ids), minlength=7)
    distance = np.linalg.norm(height_dist-new_heihgt_dist)
    min_distance = 999999
    # iter_count = 1000
    wave = wfc.wave_from_id(new_ids)
    target_error = 1
    count = 0
    print("start optimizing...")
    while distance > target_error:
        # iter_count -= 1
        new_ids,new_img = wfc.mutate(base_wave=wave, out_img=False, iter_count=2, new_weight=162.0)
        new_heihgt_dist = np.bincount(tilemap2heightmap(new_ids), minlength=7)
        distance = np.linalg.norm(height_dist-new_heihgt_dist)
        count+=1
        if distance < min_distance:
            min_distance = distance
            wave = wfc.wave_from_id(new_ids)
            print(distance)
            count = 0
        if count > 1000:
            new_ids,new_img = wfc.generate(out_img=True) 
            wave = wfc.wave_from_id(new_ids)
            min_distance = 999999
    print("done, saving...")
    unity3d_env = WFCUnity3DEnv(file_name=file_name)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unity3d_env.set_wave(new_ids)
    img = unity3d_env.render_in_unity(camera_index=0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(f"rendering and saving image")
    img_out_path = os.path.join(img_out_folder, f"same_height_easy_{time_stamp}.png")
    cv2.imwrite(img_out_path, img_bgr)
    json_out_path = os.path.join(json_out_folder, f"same_height_easy_{time_stamp}.json")
    tileid_to_json(new_ids, save_path=json_out_path)
finally:
    gamename = os.path.basename(file_name)
    print("All finish, killing game...")
    if args.os == 'windows':
        os.system(f"taskkill /im {gamename} /f")
    else:
        os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")