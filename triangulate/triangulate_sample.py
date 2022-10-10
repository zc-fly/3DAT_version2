import json
from calibrater_utility import CalibraterUtility
from triangulation_utility import TriangulationUtility


class triangulate():
	"""
	load camera model and triangulate keypoints
	"""

	# load jason file
	with open('C:/mycode/camera_calib/box-calibration-master/camera_model.json') as camera_json:
		camera_model = json.load(camera_json)

	cam_model_dict = CalibraterUtility.setup_cameras(camera_model)

	# Load the desired keypoints
	# joint_path = sys.argv[1]
	# meta_path = sys.argv[1]  # sys.argv[2]
	# joint_path = '/home/otg/Desktop/muhammad/Box_Config_E2E/biocore_livetest/run_data'
	# meta_path = '/home/otg/Desktop/muhammad/Box_Config_E2E/biocore_livetest/'
	cam_id_list = [0, 1, 2, 3]
	# offset_dict
	tc_offset_dict = [0, 0, 0, 0]

	# triang_list_dict, names = TriangulationUtility.prep_joints(joint_path, cam_id_list, tc_offset_dict)
	triang_list_dict = {}
	names = []
	final_triang_jnts, num_frms, skip_lst, pre_triang_joints = TriangulationUtility.triangulate_joints(triang_list_dict,
																									   names, camera_model,
																									   cam_model_dict)

	TriangulationUtility.write(final_triang_jnts, num_frms, skip_lst)


# CalibraterUtility.reproject_joints(meta_path, final_triang_jnts, pre_triang_joints, names, cam_model_dict, tc_offset_dict, num_frms, cam_dict)


if __name__ == '__main__':
	triangulate()
