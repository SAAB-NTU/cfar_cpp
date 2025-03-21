from utils.oculus_utils.read_oculusping import OculusFLSPing, OculusPing_generator
from utils.oculus_utils.visualize_oculusping import OculusPing2Img
from utils.nb_viewing_tools import pyplot_fig
from utils.watercol_to_cartesian_transforms import omsf_watercol_to_cartesian, scipy_skimage_watercol_to_cartesian
import matplotlib.pyplot as plt

example_file = "./20250219_201133_legacy.oculus"
folder="./20250219_201133/"
ping_gen = OculusPing_generator(example_file, verbose=False)
ping_samples = [ping for ping in ping_gen]

print("Extracted total of {} pings from sample {} file".format(len(ping_samples), example_file))


for example_ping in ping_samples[:]:
	try:
	#example_ping = ping_samples[0]
		sample_watercol = example_ping.get_acousticImg_matrix()
		sample_pingconfig = example_ping.get_pingConfig()
		print("'Config' information from sample ping:")
		#print(sample_watercol.shape)
		print(sample_pingconfig)
		
		#pyplot_fig(sample_watercol, cmap="hot", title="Example 'Watercolumn' polar-coordinate image from Oculus Pings")
		plt.imsave(folder+"polar/"+str(sample_pingconfig["pingStart_time"])+".png",sample_watercol)
		sonar_range, range_res = sample_pingconfig['user-stop_range'], sample_pingconfig['range_resolution']
		sonar_bearings = example_ping.get_brgTable_val()

		sample_cartesian = omsf_watercol_to_cartesian(sample_watercol, sonar_range, range_res=range_res, bearings_deg=sonar_bearings)
		plt.imsave(folder+"cartesian/"+str(sample_pingconfig["pingStart_time"])+".png",sample_cartesian)
		#pyplot_fig(sample_cartesian, cmap="hot", title="Example Cartesian represntation of Oculus Ping") 
	except:
		print("error")
