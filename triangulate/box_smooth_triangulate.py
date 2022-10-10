
import pandas as pd 
import numpy as np 
import cv2
import time
import matplotlib.pyplot as plt 
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline,BSpline
from scipy.signal import medfilt, sosfilt, hilbert, filtfilt, ellip, butter


def overlay(bboxes, joints, video_capture, frm_lst):

	# Associating key points with indices.
	TOP_HEAD_INDEX = 0
	BOTTOM_NECK_INDEX = 1
	RIGHT_SHOULDER_INDEX = 2
	RIGHT_ELBOW_INDEX = 3
	RIGHT_WRIST_INDEX = 4
	LEFT_SHOULDER_INDEX = 5
	LEFT_ELBOW_INDEX = 6
	LEFT_WRIST_INDEX = 7
	RIGHT_HIP_INDEX = 8
	RIGHT_KNEE_INDEX = 9
	RIGHT_ANKLE_INDEX = 10
	RIGHT_HEEL_INDEX = 11
	RIGHT_TOE_INDEX = 12
	LEFT_HIP_INDEX = 13
	LEFT_KNEE_INDEX = 14
	LEFT_ANKLE_INDEX = 15
	LEFT_HEEL_INDEX = 16
	LEFT_TOE_INDEX = 17

	# Color for the skeletal overlay.
	color = (0, 255, 0)



	# Allow for the frame buffer to fill.
	time.sleep(1.0)

	video_writer = cv2.VideoWriter(
		"video_3.mp4",
		cv2.VideoWriter_fourcc(*"mp4v"),
		video_capture.get(cv2.CAP_PROP_FPS),
		(848, 480)
	)

	# print(frm_lst)


	done = False
	i = 1

	
	while not done:

		# Read the frame.
		ret, current_frame = video_capture.read()
		# plt.imshow(current_frame)
		# plt.scatter(11.6, 11.7)
		# plt.show()

		if not ret:
			break
		fig = plt.figure()
		plt.imshow(current_frame)
		#print('i', i)
		#print('len', np.where(frm_lst == i))
		if len(np.where(frm_lst == i)[0]) > 0:
			ind = np.where(frm_lst == i)[0][0]

			x1 = bboxes.iloc[ind]['x']
			y1 = bboxes.iloc[ind]['y']
			x2 = x1 + bboxes.iloc[ind]['w']
			y2 = y1 + bboxes.iloc[ind]['h']

			athlete_prediction = np.array(joints.iloc[ind].tolist())[1::]

			# Round and reshape the key points to make it easier to draw them on the overlaid frame.
			#athlete_prediction = [int(round(float(x))) for x in athlete_prediction]
			points = []

			for ml in range(18):
				i1 = 2 * ml
				i2 = 2 * ml + 1
				points.append((athlete_prediction[i1], athlete_prediction[i2]))
			points.append((athlete_prediction[0],athlete_prediction[1]))

			
			plt.plot([points[LEFT_HIP_INDEX][0], points[RIGHT_HIP_INDEX][0]], [points[LEFT_HIP_INDEX][1], points[RIGHT_HIP_INDEX][1]], 'C1')
			plt.plot([points[LEFT_SHOULDER_INDEX][0], points[RIGHT_SHOULDER_INDEX][0]], [points[LEFT_SHOULDER_INDEX][1], points[RIGHT_SHOULDER_INDEX][1]], 'C1')
			plt.plot([points[RIGHT_HIP_INDEX][0], points[RIGHT_SHOULDER_INDEX][0]], [points[RIGHT_HIP_INDEX][1], points[RIGHT_SHOULDER_INDEX][1]], 'C1')
			plt.plot([points[LEFT_HIP_INDEX][0], points[LEFT_SHOULDER_INDEX][0]], [points[LEFT_HIP_INDEX][1], points[LEFT_SHOULDER_INDEX][1]], 'C1')
			plt.plot([points[BOTTOM_NECK_INDEX][0], points[TOP_HEAD_INDEX][0]], [points[BOTTOM_NECK_INDEX][1], points[TOP_HEAD_INDEX][1]], 'C1')
			
			#plt.show()
			# Draw the torso.
			# cv2.line(current_frame, points[LEFT_HIP_INDEX], points[RIGHT_HIP_INDEX], color, 2)
			# cv2.line(current_frame, points[LEFT_SHOULDER_INDEX], points[RIGHT_SHOULDER_INDEX], color, 2)
			# cv2.line(current_frame, points[RIGHT_HIP_INDEX], points[RIGHT_SHOULDER_INDEX], color, 2)
			# cv2.line(current_frame, points[LEFT_HIP_INDEX], points[LEFT_SHOULDER_INDEX], color, 2)
			# cv2.line(current_frame, points[BOTTOM_NECK_INDEX], points[TOP_HEAD_INDEX], color, 2)


			plt.plot([points[LEFT_KNEE_INDEX][0], points[LEFT_HIP_INDEX][0]], [points[LEFT_KNEE_INDEX][1], points[LEFT_HIP_INDEX][1]], 'C1')
			plt.plot([points[LEFT_ANKLE_INDEX][0], points[LEFT_KNEE_INDEX][0]], [points[LEFT_ANKLE_INDEX][1], points[LEFT_KNEE_INDEX][1]], 'C1')
			plt.plot([points[RIGHT_KNEE_INDEX][0], points[RIGHT_HIP_INDEX][0]], [points[RIGHT_KNEE_INDEX][1], points[RIGHT_HIP_INDEX][1]], 'C1')
			plt.plot([points[RIGHT_ANKLE_INDEX][0], points[RIGHT_KNEE_INDEX][0]], [points[RIGHT_ANKLE_INDEX][1], points[RIGHT_KNEE_INDEX][1]], 'C1')

			# plt.show()
			# Draw the legs.
			# cv2.line(current_frame, points[LEFT_KNEE_INDEX], points[LEFT_HIP_INDEX], color, 2)
			# cv2.line(current_frame, points[LEFT_ANKLE_INDEX], points[LEFT_KNEE_INDEX], color, 2)
			# cv2.line(current_frame, points[RIGHT_KNEE_INDEX], points[RIGHT_HIP_INDEX], color, 2)
			# cv2.line(current_frame, points[RIGHT_ANKLE_INDEX], points[RIGHT_KNEE_INDEX], color, 2)

			plt.plot([points[LEFT_SHOULDER_INDEX][0], points[LEFT_ELBOW_INDEX][0]], [points[LEFT_SHOULDER_INDEX][1], points[LEFT_ELBOW_INDEX][1]], 'C1')
			plt.plot([points[LEFT_ELBOW_INDEX][0], points[LEFT_WRIST_INDEX][0]], [points[LEFT_ELBOW_INDEX][1], points[LEFT_WRIST_INDEX][1]], 'C1')
			plt.plot([points[RIGHT_SHOULDER_INDEX][0], points[RIGHT_ELBOW_INDEX][0]], [points[RIGHT_SHOULDER_INDEX][1], points[RIGHT_ELBOW_INDEX][1]], 'C1')
			plt.plot([points[RIGHT_ELBOW_INDEX][0], points[RIGHT_WRIST_INDEX][0]], [points[RIGHT_ELBOW_INDEX][1], points[RIGHT_WRIST_INDEX][1]], 'C1')

			# Draw the arms.
			# cv2.line(current_frame, points[LEFT_SHOULDER_INDEX], points[LEFT_ELBOW_INDEX], color, 2)
			# cv2.line(current_frame, points[LEFT_ELBOW_INDEX], points[LEFT_WRIST_INDEX], color, 2)
			# cv2.line(current_frame, points[RIGHT_SHOULDER_INDEX], points[RIGHT_ELBOW_INDEX], color, 2)
			# cv2.line(current_frame, points[RIGHT_ELBOW_INDEX], points[RIGHT_WRIST_INDEX], color, 2)

			plt.plot([points[LEFT_ANKLE_INDEX][0], points[LEFT_TOE_INDEX][0]], [points[LEFT_ANKLE_INDEX][1], points[LEFT_TOE_INDEX][1]], 'C1')
			plt.plot([points[LEFT_ANKLE_INDEX][0], points[LEFT_HEEL_INDEX][0]], [points[LEFT_ANKLE_INDEX][1], points[LEFT_HEEL_INDEX][1]], 'C1')
			plt.plot([points[RIGHT_ANKLE_INDEX][0], points[RIGHT_TOE_INDEX][0]], [points[RIGHT_ANKLE_INDEX][1], points[RIGHT_TOE_INDEX][1]], 'C1')
			plt.plot([points[RIGHT_ANKLE_INDEX][0], points[RIGHT_HEEL_INDEX][0]], [points[RIGHT_ANKLE_INDEX][1], points[RIGHT_HEEL_INDEX][1]], 'C1')

			# Draw the feet.
			# cv2.line(current_frame, points[LEFT_ANKLE_INDEX], points[LEFT_TOE_INDEX], color, 2)
			# cv2.line(current_frame, points[LEFT_ANKLE_INDEX], points[LEFT_HEEL_INDEX], color, 2)
			# cv2.line(current_frame, points[RIGHT_ANKLE_INDEX], points[RIGHT_TOE_INDEX], color, 2)
			# cv2.line(current_frame, points[RIGHT_ANKLE_INDEX], points[RIGHT_HEEL_INDEX], color, 2)

			# Display the frame number.
			cv2.putText(
				current_frame,
				str(i),
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				(0, 0, 0),
				2,
				cv2.LINE_AA
			)

			overlay = current_frame.copy()
			output = current_frame.copy()


			cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
			cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
			current_frame = output

			# Write the overlaid frame.
			video_writer.write(current_frame)

		else:

			video_writer.write(current_frame)

		#plt.show()
		fig.savefig('./tmp/'+str(i)+'.png')
		plt.close('all')


		i += 1




def get_smooth_joint(bbox_fname, jnts_src):
	
	
	tmp = pd.read_csv(bbox_fname)
	frames = np.array([int(x) for x in tmp['frame'].tolist()])
	tmp2 = pd.read_csv(jnts_src)


	names = [nm for nm in tmp2.columns[1::]]

	#print(names)
	data = {}
	data['frame'] = (tmp['frame']).tolist()
	for n in names:
		vals = np.array(tmp2[n].tolist())
		# dom = np.linspace(0, len(vals)-1, len(vals))
		dom = np.array((tmp['frame']).tolist())
		gt_inds = np.where(vals > 0)[0]
		b, a = ellip(4, 0.01, 10, 0.2)  # Filter to be applied.
		new = filtfilt(b,a, vals[gt_inds], method="gust")
		#new = vals[gt_inds]
		vals_spl = BSpline(dom[gt_inds], vals[gt_inds], k=2)
		#final_vals = vals_spl(dom)
		#final_vals = lp_filter(vals[gt_inds])

		new_dat = np.zeros(vals.shape)
		new_dat[gt_inds] = new
		# new_dat = vals_spl(dom)

		b_inds = np.where(np.abs(new_dat - vals) > 10)[0]

		
		data[n] = new_dat.tolist()


	new_df = pd.DataFrame(data, columns = [nm for nm in tmp2.columns])

	#new_nm = bbox_fname.split('/')[-1].split('.')[0]

	#new_df.to_csv(, index=False)
	#video_capture = cv2.VideoCapture(vid_src)

	#overlay(tmp, new_df, video_capture, frames)
	return new_df







