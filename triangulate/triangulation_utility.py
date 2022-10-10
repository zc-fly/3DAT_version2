import sys
import os
import numpy as np
from joblib import Parallel, delayed

from .calibrater_utility import CalibraterUtility
from .camera_interaction_toolkit import operations
from .box_smooth_triangulate import get_smooth_joint

class TriangulationUtility:

	@staticmethod
	def triangulate_point(cam_model_dict, points, accp_lst):

		# print(points)
		rays = []
		for i, k in enumerate(cam_model_dict.keys()):
			
			if k in accp_lst:
				cm = cam_model_dict[k]
				ray_start = cm.position
				ray_end = cm.image_to_world(points[i], vector_length=1)
				rays.append((ray_start, ray_end - ray_start))

		
		position = operations.triangulate_rays(rays)

		return position
		#print(position)

	@staticmethod
	def triangulate_ground_plane(cam_model_dict, grnd_plane_pt_dict, accp_lst):

		d_corners = []
		for j in range(4):
			tmp_lst = []
			for k in grnd_plane_pt_dict.keys():
				tmp_lst.append(grnd_plane_pt_dict[k][j])

			res = CalibraterUtility.triangulate_point(cam_model_dict, tmp_lst, accp_lst)
			d_corners.append(res)
		return np.array(d_corners)

	@staticmethod
	def load_joint_csv(joint_path, cam_num):
		i = cam_num
		kp_df = get_smooth_joint(os.path.join(joint_path, '2d_cam_'+str(i+1)+'_bounding_boxes.csv'),
			os.path.join(joint_path, '2d_cam_'+str(i+1)+'_key_points.csv'))

		return kp_df

	@staticmethod
	def prep_joints(joint_path, cam_id_lst, tc_offset_dict):

		triang_list_dict = {}
		names = []
		for i in cam_id_lst:
			triang_list_dict[i] = {}
			kp_df = TriangulationUtility.load_joint_csv(joint_path, i)

			frames = np.array(kp_df['frame'].tolist())
			inds = np.where(frames >= tc_offset_dict[i])[0]

			names = [nm for nm in kp_df.columns[1::]]

			for n in names:
				triang_list_dict[i][n] = np.array(kp_df[n].tolist())[inds]

		return triang_list_dict, names

	@staticmethod
	def triangulate_joints(triang_list_dict, names, cam_calib_dict, cam_model_dict):
		
		def triangulate_point(tupl):
			# print(points)
			points, accp_lst = tupl
			rays = []
			for i, k in enumerate(cam_model_dict.keys()):
				
				if accp_lst[i]>0:
					cm = cam_model_dict[k]
					ray_start = cm.position
					ray_end = cm.image_to_world(points[i], vector_length=1)
					rays.append((ray_start, ray_end - ray_start))

			position = operations.triangulate_rays(rays)

			camera_num = len(cam_model_dict)
			if len(rays) == camera_num:
				distances = [operations.distance_from_point_to_ray(position, ray_start, ray_direction)
								for ray_start, ray_direction in rays]
				if any(distance > .07 for distance in distances):
					max_error_idx = np.argmax(distances)
					rays_subset = [ray for i, ray in enumerate(rays) if i != max_error_idx]
					position = operations.triangulate_rays(rays_subset)

			cm = cam_model_dict['Camera1']
			imagepx = cm.world_to_image(position)

			return position
		
		final_triang_jnts = {}
		camera_num = len(cam_model_dict)
		for k in range(len(names)):

			res = np.zeros((2, camera_num ))

			cam_use_dict = {}
			for i, cam in enumerate(cam_calib_dict.keys()):
				cam_use_dict[i] = 0
				tmp = triang_list_dict[i][k,:]
				res[0:2, i] = tmp

				pts = res[:,i].reshape((1,1,2))

				CameraMatrix = (np.array(cam_calib_dict[cam]['CameraIntrinsic'])).reshape((3, 3))
				DistCoeff = np.array(cam_calib_dict[cam]['DistCoeff'])
				h = cam_calib_dict[cam]['FrameHeight']
				w = cam_calib_dict[cam]['FrameWidth']
				try:
					undis_pts = CalibraterUtility.undistort_points(CameraMatrix, DistCoeff, pts,
															   [int(h), int(w)])
				except Exception as e:
					print(e)

				res[:,i] = undis_pts.reshape((1, 2))

				cam_use_dict[i] = 1

			accp_lst = [cam_use_dict[0],
						cam_use_dict[1],
						cam_use_dict[2],
						cam_use_dict[3]]
			# if len(np.where(accp_lst==0)[0]) > 2:
			# 	skip_lst.append(ll)
			poiints = [res[:,0], res[:,1],res[:,2],res[:,3]]

			res = triangulate_point((poiints,accp_lst))
			final_triang_jnts[names[k]] = [res.copy(), poiints[0]]

		return final_triang_jnts


	@staticmethod
	def triangulate_SVDmethod(triang_list_dict, names, cam_calib_dict, cam_model_dict):

		def triangulate_point_svd(tupl):
			# print(points)
			points, accp_lst = tupl
			n = len(points)
			A = np.zeros((2*n, 4))
			for i, k in enumerate(cam_model_dict.keys()):
				cm = cam_model_dict[k]
				P = cm.projection
				A[2*i] = points[i][0] * P[2] - P[0]
				A[2*i+1] = points[i][1] * P[2] - P[1]
			_, _, vt = np.linalg.svd(A)
			position = vt[3,0:3] / vt[3,3]
			return position

		final_triang_jnts = {}
		camera_num = len(cam_model_dict)
		for k in range(len(names)):

			res = np.zeros((2, camera_num))

			cam_use_dict = {}
			for i, cam in enumerate(cam_calib_dict.keys()):
				cam_use_dict[i] = 0
				tmp = triang_list_dict[i][k, :]
				res[0:2, i] = tmp

				pts = res[:, i].reshape((1, 1, 2))

				CameraMatrix = (np.array(cam_calib_dict[cam]['CameraIntrinsic'])).reshape((3, 3))
				DistCoeff = np.array(cam_calib_dict[cam]['DistCoeff'])
				h = cam_calib_dict[cam]['FrameHeight']
				w = cam_calib_dict[cam]['FrameWidth']
				try:
					undis_pts = CalibraterUtility.undistort_points(CameraMatrix, DistCoeff, pts,
																   [int(h), int(w)])
				except Exception as e:
					print(e)

				res[:, i] = undis_pts.reshape((1, 2))

				cam_use_dict[i] = 1

			accp_lst = [cam_use_dict[0],
						cam_use_dict[1],
						cam_use_dict[2],
						cam_use_dict[3]]
			# if len(np.where(accp_lst==0)[0]) > 2:
			# 	skip_lst.append(ll)
			poiints = [res[:, 0], res[:, 1], res[:, 2], res[:, 3]]

			res = triangulate_point_svd((poiints, accp_lst))
			final_triang_jnts[names[k]] = [res.copy(), poiints[0]]

		return final_triang_jnts


	@staticmethod
	def write(jnt_dict, num_frms, skip_lst, output_path):


		file1 = open(f"{output_path}/box-config-joints.txt","w")#write mode


		stringy = ''

		jnt_order = ['ankler', 'kneer', 'hipr', 'hipl', 'kneel', 'anklel', 'pelv', 'neck',
					 'nape', 'head', 'wristr', 'elbowr', 'shoulderr', 'shoulderl', 'elbowl',
					 'wristl', 'toel', 'toer', 'heell', 'heelr']
		
		for i in range(num_frms):

			if i not in skip_lst:
				toy_dat = []
				for jnt in jnt_order:
					if jnt == 'pelv':
						toy_dat += ((jnt_dict['hipr'][i] + jnt_dict['hipl'][i])/2).tolist()
					elif jnt == 'nape':
						toy_dat += (jnt_dict['head'][i] - (jnt_dict['neck'][i])/2).tolist()
					else:
						toy_dat += jnt_dict[jnt][i].tolist()
				
				stringy += 'mpii_20, player_1, 2_F' + str(i).zfill(4) + ', ' + str(i) + ': ' + ','.join([str(x) for x in toy_dat])
				stringy = stringy.strip()
				stringy += ',\n'

			# print(','.join([str(x) for x in toy_dat.flatten()]))
		file1.write(stringy)
		
		file1.close()