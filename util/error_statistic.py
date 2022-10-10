import open3d as o3d  #note: open3d==0.8
import numpy as np
import pickle
import time
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
    # connect_lines = [[0,1],[1,2],[0,2],[1,3],[2,4],[0,5],[0,6],[5,6],[5,11],[6,12],[11,12],[5,7],[7,9],[6,8],[8,10],[11,13],[12,14],[13,15],[14,16]]
    #
    # groundTruth = [50,64,50,110,110,280,280,360,440,440,290,265,280,265,280,430,430,470,470] #mm
    # calculateValue = []
    #
    # with open("D:/intel/sfm/zc_sfm_01/3DkeyPoints.pickle", 'rb') as file:
    #     triout = pickle.load(file)
    #
    # for frame in triout:
    #     frameBone = []
    #     for bone in connect_lines:
    #         Bone = frame[bone[0],:] - frame[bone[1],:]
    #         Bone = np.sqrt(np.sum(np.square(Bone)))
    #         frameBone.append(Bone)
    #     calculateValue.append(frameBone)
    #
    # sum = 0
    # for i in calculateValue:
    #     sum += i[1]
    #
    # eyedis = sum/len(calculateValue)
    #
    # Scale = groundTruth[1] / eyedis
    # calculateValue = Scale * np.array(calculateValue)
    # error = calculateValue - np.array(groundTruth)
    # x = np.linspace(1,19,19)
    # # for i in range(len(calculateValue)):
    # #     plt.scatter(x, error[i, :])
    # # plt.show()
    # plt.ion()
    # for i in range(len(calculateValue)):
    #     plt.cla
    #     plt.scatter(x, error[i, :])
    #     plt.pause(0.1)
    # plt.ioff()
    # plt.show()



    names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
    connect_lines = [[0,1],[1,2],[0,2],[1,3],[2,4],[0,5],[0,6],[5,6],[5,11],[6,12],[11,12],[5,7],[7,9],[6,8],[8,10],[11,13],[12,14],[13,15],[14,16]]

    groundTruth = [50,64,50,110,110,280,280,360,600,600,290,265,280,265,280,430,430,470,470] #mm
    calculateValue = []

    with open("D:/intel/sfm/zc_sfm_01/3DkeyPoints.pickle", 'rb') as file:
        triout = pickle.load(file)

    for frame in triout:
        frameBone = []
        for bone in connect_lines:
            Bone = frame[bone[0],:] - frame[bone[1],:]
            a = np.square(Bone)
            b = np.sum(a)
            c = np.sqrt(b)
            Bone = np.sqrt(np.sum(np.square(Bone)))
            frameBone.append(Bone)
        calculateValue.append(frameBone)

    calculateValue = np.array(calculateValue)
    x = np.linspace(1,19,19)
    plt.ion()
    count = 0
    for i in range(len(calculateValue)):
        if count%30==0:
            Scale = groundTruth[16] / calculateValue[i,16]
            see = Scale * calculateValue[i]
            error = Scale * calculateValue[i] - np.array(groundTruth)
            plt.cla
            plt.scatter(x, error,s = 10)
            plt.pause(0.1)
        count+=1
    plt.ioff()
    plt.show()