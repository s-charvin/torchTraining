# https://github.com/IoannisKansizoglou/Iemocap-preprocess
###################### LIBRARIES ######################
import cv2
import numpy as np
import os
import os.path
from utils.filefolderTool import create_folder
# Threshold for keeping a center movement per frame
c_threshold = 40
# Threshold for keeping a center vs other
d_threshold = 100000
# Values for first cropping
y1, y2, y3, y4, x1, x2, x3, x4 = (
    130, 230, 140, 240, 120, 240, 500, 630)  # w,h
# Final size of cropped image
width, height = (240, 240)
###################### FUNCTIONS #####################


### Function to crop image tracking the face ###
def crop_face_image(cascPATH, image, left_precenter, right_precenter, speaker):

    # Set face tracking type
    cascPATH = cascPATH
    face_cascade = cv2.CascadeClassifier(cascPATH)
    # First crop of image to simplify face-detection
    img = image.copy()
    img1 = img[y1:y2, x1:x2]
    img2 = img[y3:y4, x3:x4]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Detect faces (-eyes) in the image
    left_faces = face_cascade.detectMultiScale(
        gray1, scaleFactor=1.01, minNeighbors=5)
    right_faces = face_cascade.detectMultiScale(
        gray2, scaleFactor=1.01, minNeighbors=5)
    # Track left speaker
    left_center, left_index = select_window(left_faces, left_precenter)
    # Track right speaker
    right_center, right_index = select_window(right_faces, right_precenter)
    # Calculate left crop rectangle
    left_a = left_center[0] - int(width/2)
    left_b = left_center[1] - int(height/2)
    # Calculate right crop rectangle
    right_a = right_center[0] - int(width/2)
    right_b = right_center[1] - int(height/2)

    # Select speaker to crop
    if speaker == 'L':
        # Crop image
        left_a += x1
        left_b += y1
        image = image[left_b:left_b+width, left_a:left_a+height]

    elif speaker == 'R':
        # Crop image
        right_a += x3
        right_b += y3
        image = image[right_b:right_b+width, right_a:right_a+height]
        #image = image[int((y3+y4)/2)-int(width/2):int((y3+y4)/2)-int(width/2)+width, int((x3+x4)/2)-int(height/2):int((x3+x4)/2)-int(height/2)+height]

    else:

        image = np.array([0, 0])

    # Return cropped image and next precenters
    return image, left_center, right_center


# Function to select the best window before cropping
def select_window(faces, precenter):

    # Case[1] of detecting no 'face'
    if np.shape(faces)[0] == 0:
        # Proposed center
        center = precenter
        # False value for index in faces of selected window
        index = -1

    else:

        # Index in faces of selected window
        index = 0
        # Case[2] of detecting many 'faces'
        if np.shape(faces)[0] > 1:

            # Starting default distance
            dmin = d_threshold
            i = 0

            # Decide which to keep
            for (x, y, w, h) in faces:

                # Compute center
                xc = int(round((2*x+w)/2))
                yc = int(round((2*y+h)/2))
                d = np.linalg.norm(np.array([xc, yc])-np.array(precenter))
                # Change appropriately min and index
                if d < dmin:
                    dmin = d
                    index = i

                i += 1

            # Take values for proposed center
            index = int(index)
            x, y, w, h = faces[index]

        # Case[3] of detecting exactly one 'face'
        else:

            # Take values for proposed center
            x, y, w, h = faces[0]
            xc = int(round((2*x+w)/2))
            yc = int(round((2*y+h)/2))
            dmin = np.linalg.norm(np.array([xc, yc])-np.array(precenter))

        # Proposed centre
        xc = int(round((2*x+w)/2))
        yc = int(round((2*y+h)/2))
        # Check distance with precenter threshold
        if dmin < c_threshold:
            # Proposed center is accepted
            center = [xc, yc]
        else:
            # Proposed center is discarded, keep precenter
            center = precenter

        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # Compute big axis of face-detection rectangle
        #s1 = np.array([x, y])
        #s2 = np.array([x+w, y+h])
        # print(np.linalg.norm(s1-s2))

    if precenter == [0, 0]:
        if np.shape(faces)[0] > 0:
            x, y, w, h = faces[0]
            xc = int(round((2*x+w)/2))
            yc = int(round((2*y+h)/2))
            center = [xc, yc]
        else:
            center = [int((x2-x1)/2), int((y2-y1)/2)]

    return center, index


def capture_face_video(cascPATH, in_path, out_path, start_time, end_time, speaker):

    # First precenters don't exist
    left_precenter = [0, 0]
    right_precenter = [0, 0]

    if os.path.isfile(in_path):
        # Playing video from file
        cap = cv2.VideoCapture(in_path)
        # 读取视频帧率
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        # 设置写入视频的编码格式
        fourcc = cv2.VideoWriter_fourcc(
            *'XVID')
        # 设置写视频的对象

        videoWriter = cv2.VideoWriter(
            out_path, fourcc, fps_video, (width, height))
        count = 0
        while (cap.isOpened()):
            success, frame = cap.read()
            if success == True:
                count += 1
                # 截取相应时间内的视频信息
                if(count > (start_time * fps_video) and count <= (end_time * fps_video)):
                    # 截取相应的视频帧
                    frame, left_precenter, right_precenter = crop_face_image(cascPATH,
                                                                             frame, left_precenter, right_precenter, speaker)
                    # 将图片写入视频文件
                    videoWriter.write(frame)
                if(count == (end_time * fps_video)):
                    break
            else:
                # 写入视频结束

                break
    # When everything is done, release the capture
    videoWriter.release()
    cap.release()
    return out_path


def capture_video(in_path, out_path, start_time, end_time, speaker):

    if os.path.isfile(in_path):
        # Playing video from file
        cap = cv2.VideoCapture(in_path)
        # 读取视频帧率
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        # 设置写入视频的编码格式
        fourcc = cv2.VideoWriter_fourcc(
            *'XVID')
        # 设置写视频的对象

        videoWriter = cv2.VideoWriter(
            out_path, fourcc, fps_video, (width, height))
        count = 0
        # 设置裁剪窗口, [宽起始点,高起始点,宽度,高度]
        window_l = [5, 120, 350, 240]
        window_r = [365, 120, 350, 240]
        while (cap.isOpened()):
            success, frame = cap.read()  # size: h,w
            if success == True:
                count += 1
                # 截取相应时间内的视频信息
                if(count > (start_time * fps_video) and count <= (end_time * fps_video)):
                    # 截取相应的视频帧

                    if speaker == 'L':
                        frame = frame[window_l[1]:window_l[4],
                                      window_l[0]:window_l[3]]
                    elif speaker == 'R':
                        frame = frame[window_r[1]:window_r[4],
                                      window_r[0]:window_r[3]]
                    # 将图片写入视频文件
                    videoWriter.write(frame)
                if(count == (end_time * fps_video)):
                    break
            else:
                # 写入视频结束

                break
    # When everything is done, release the capture
    videoWriter.release()
    cap.release()
    return out_path


# Control runtime
if __name__ == '__main__':
    capture_face_video()
