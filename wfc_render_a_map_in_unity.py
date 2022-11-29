from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
import fastwfc
wfc = fastwfc.XLandWFC("samples.xml")
unity3d_env = WFCUnity3DEnv()
seed,_ = wfc.generate(out_img=False)
wave = wfc.wave_from_id(seed)
unity3d_env.set_wave(wave=wave)
unity3d_env.render_in_unity()