from utils.oculus_utils.read_oculusping import OculusFLSPing, OculusPing_generator
from utils.oculus_utils.visualize_oculusping import OculusPing2Img
from utils.nb_viewing_tools import pyplot_fig
from utils.watercol_to_cartesian_transforms import omsf_watercol_to_cartesian, scipy_skimage_watercol_to_cartesian
import matplotlib.pyplot as plt

example_file = "./Oculus_20220520_134830+Surface+tgt_range_3_5m+Vert_View+M1200D+1200kHz_Freq+maxrange_5m+gain_50.oculus"
ping_gen = OculusPing_generator(example_file, verbose=False)
ping_samples = [ping for ping in ping_gen]

print("Extracted total of {} pings from sample {} file".format(len(ping_samples), example_file))

example_ping = ping_samples[0]
sample_watercol = example_ping.get_acousticImg_matrix()
sample_pingconfig = example_ping.get_pingConfig()
print("'Config' information from sample ping:")
print(sample_pingconfig)
print(sample_watercol)
pyplot_fig(sample_watercol, cmap="hot", title="Example 'Watercolumn' polar-coordinate image from Oculus Pings")

sonar_range, range_res = sample_pingconfig['user-stop_range'], sample_pingconfig['range_resolution']
sonar_bearings = example_ping.get_brgTable_val()

sample_cartesian = omsf_watercol_to_cartesian(sample_watercol, sonar_range, range_res=range_res, bearings_deg=sonar_bearings)
pyplot_fig(sample_cartesian, cmap="hot", title="Example Cartesian represntation of Oculus Ping") 
