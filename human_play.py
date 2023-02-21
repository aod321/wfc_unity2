import numpy as np
# from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
from WFCUnity3DEnv_fastwfc_nodepair import WFCUnity3DEnv
import fastwfc
import cv2
import pygame


_SIZE = [800, 600]
_FRAMES_PER_SEC = 50
_FRAME_DELAY_MS = int(1000.0 // _FRAMES_PER_SEC)

_ACTION_NOTHING = 7
# _ACTION_LOOKUP = 3
# _ACTION_LOOKDOWN = 4
_ACTION_FORWARD = 0
_ACTION_BACKWARD = 1
_ACTION_LEFT = 2
_ACTION_RIGHT = 3
_ACTION_LOOKLEFT = 4
_ACTION_LOOKRIGHT = 5
# ip = "192.168.123.109"
ip = "0.0.0.0"

def main():
    pygame.init()
    wfc_size = 9
    port = 30051
    timeout = 10
    print("create and join world")
    env = WFCUnity3DEnv(host=ip, port=port, wfc_size=wfc_size, return_all=False, camera_size=_SIZE)
    wfc = fastwfc.XLandWFC("samples.xml")
    wave,_ = wfc.generate(out_img=False)
    env.render_in_unity()
    env.set_wave(wave=wave)
    env.render_in_unity()
    env.reset()
    window_surface = pygame.display.set_mode(_SIZE, 0, 32)
    pygame.display.set_caption("world")
    window_surface.fill((128, 128, 128))
    with env:
        keep_running = True
        while keep_running:
            requested_action = _ACTION_NOTHING
            is_jumping = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keep_running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        requested_action = _ACTION_LEFT
                    elif event.key == pygame.K_RIGHT:
                        requested_action = _ACTION_RIGHT
                    if event.key == pygame.K_UP:
                        requested_action = _ACTION_FORWARD
                    elif event.key == pygame.K_DOWN:
                        requested_action = _ACTION_BACKWARD
                    elif event.key == pygame.K_q:
                        requested_action = _ACTION_LOOKLEFT
                    elif event.key == pygame.K_e:
                        requested_action = _ACTION_LOOKRIGHT
                    elif event.key == pygame.K_r:
                        wave,_ = wfc.generate(out_img=False)
                        env.set_wave(wave=wave)
                        env.render_in_unity()
                    elif event.key == pygame.K_SPACE:
                        is_jumping = 1
                        env.reset()
                    elif event.key == pygame.K_ESCAPE:
                        keep_running = False
                        break
            try:
                image_obs, reward, done, _ = env.step(requested_action)
                # ortate image -90 degree
                image = cv2.rotate(image_obs, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                image = pygame.surfarray.make_surface(image)
                image = pygame.transform.flip(image, False, True)
                window_surface.blit(image, (0, 0))
                pygame.display.update()
                pygame.time.wait(_FRAME_DELAY_MS)
            finally:
                pass
        
if __name__ == "__main__":
    main()
