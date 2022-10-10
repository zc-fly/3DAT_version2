#This file aim to unify human keypoints

#setting yourself human keypoints type
# Note: if changed, function below & util/viewer.py connection relationship must be changed accordinates
names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

def unify(keyPoint, locType):

    if locType==0:#hrnet
        return keyPoint

    if locType==1:#balzepose
        chooseList = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
        keyPoint = keyPoint[chooseList,:]
        return keyPoint