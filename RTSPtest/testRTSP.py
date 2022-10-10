import subprocess as sp
import cv2
import sys
import queue
import threading

# frame_queue = queue.Queue()
# rtmpUrl = "rtmp:/10.11.208.230/live"
# camera_path = 'D:/intel/sfm/zc_sfm_01/c1.mp4'  # 这是湖南台的实时直播流
#
# # 获取摄像头参数
# cap = cv2.VideoCapture(camera_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # print(fps, width, height)
#
# # ffmpeg command
# command = ['ffmpeg',
#            '-y',
#            'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',
#            '-s', "{}x{}".format(width, height),
#            '-r', str(fps),
#            '-c:v', 'libx264',
#            '-pix_fmt', 'yuv420p',
#            '-preset', 'ultrafast',
#            '-f', 'flv',
#            '-g', '5',
#            rtmpUrl]
#
#
# # 读流函数
# def Video():
#     vid = cv2.VideoCapture(camera_path)
#     if not vid.isOpened():
#         raise IOError("could't open webcamera or video")
#     while (vid.isOpened()):
#         ret, frame = vid.read()
#         # 下面注释的代码是为了防止摄像头打不开而造成断流
#         # if not ret:
#         # vid = cv2.VideoCapture(camera_path）
#         # if not vid.isOpened():
#         # raise IOError("couldn't open webcamera or video")
#         # continue
#         frame_queue.put(frame)
#
#
# def push_stream(left_x, left_y, right_x, right_y):
#     # 管道配置
#     while True:
#         if len(command) > 0:
#             p = sp.Popen(command, stdin=sp.PIPE)
#             break
#
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             if frame is not None:
#                 # 我这里出现了frame为NoneType的情况，所以判断一下
#                 image = cv2.resize(frame[int(left_x):int(right_x)][int(left_y):int(right_y)], (width, height))
#                 p.stdin.write(image.tostring())
#
#
# def run(left_x, left_y, right_x, right_y):
#     thread_video = threading.Thread(target=Video, )
#     thread_push = threading.Thread(target=push_stream, args=(left_x, left_y, right_x, right_y,))
#     thread_video.start()
#     thread_push.start()
#
#
# if __name__ == "__main__":
#
#     left_x = 1
#     left_y = 1
#     right_x = 800
#     right_y = 800
#
#     with open("zoomfile.txt", "w") as f:
#         f.write("0")
#     run(left_x, left_y, right_x, right_y)

screen = cv2.VideoCapture('rtsp://10.11.208.230:8554/video')

while True:

    #img即为
    sucess,img=screen.read()
    #转为灰度图片
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #显示摄像头
    cv2.imshow("img",gray)
    #保持画面的持续。
    k=cv2.waitKey(1)      #这里如果为0的话，就是将你目前所在的画面定定格，为其他数字比如1的时候，表示1秒后程序结束。但是由于是死循环，所以结束后马上开启，就为连续图像，
    if k == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break
    elif k==ord("s"):
        #通过s键保存图片，并退出。
        cv2.imwrite("image2.jpg",img)
        cv2.destroyAllWindows()
        break
#关闭摄像头
screen.release()
