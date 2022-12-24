import cv2
import os
import argparse
import fastwfc
from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
from utils import tileid_to_json
from datetime import datetime


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--os", type=str, required=True, default="macos", help="macos or windows")
arg_parser.add_argument("--count", type=int, required=False, default=10, help="numbers to render")
args = arg_parser.parse_args()

if args.os == "windows":
    file_name = "./windows_build/1209_windows_build/tilemap_render.exe"
else:
    file_name = "./mac_build/1209_mac_built.app/Contents/MacOS/tilemap_render"

img_out_folder = "out_images"
json_out_folder = "out_jsons"
os.makedirs(img_out_folder, exist_ok=True)
os.makedirs(json_out_folder, exist_ok=True)
try:
    wfc = fastwfc.XLandWFC("samples.xml")
    unity3d_env = WFCUnity3DEnv(file_name=file_name)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(args.count):
        seed,_ = wfc.generate(out_img=False)
        unity3d_env.set_wave(seed)
        img = unity3d_env.render_in_unity(camera_index=0)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(f"rendering and saving image {i}")
        img_out_path = os.path.join(img_out_folder, f"{time_stamp}_{i}.png")
        cv2.imwrite(img_out_path, img_bgr)
        json_out_path = os.path.join(json_out_folder, f"{time_stamp}_{i}.json")
        tileid_to_json(seed, save_path=json_out_path)

    # cv2.waitKey(0)
finally:
    gamename = os.path.basename(file_name)
    print("All finish, killing game...")
    if args.os == 'windows':
        os.system(f"taskkill /im {gamename} /f")
    else:
        os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")