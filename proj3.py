import cv2
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join
import math

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def detect_corner(frame, height, width):
    frame_0 = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    h_channel = frame_0[:, :, 0]
    cv2.imshow("h_channel image", h_channel)
    # cv2.waitKey(0)
    l_channel = frame_0[:, :, 1]
    cv2.imshow("l_channel image", l_channel)
    # cv2.waitKey(0)
    s_channel = frame_0[:, :, 2]
    cv2.imshow("s_channel image", s_channel)
    cv2.imwrite("s_channel image.jpeg", s_channel)
    # cv2.waitKey(0)

    Z = s_channel.reshape((-1, 1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((s_channel.shape))
    cv2.imshow("threshold after s_channel image", res2)
    cv2.imwrite('threshold after s_channel image.jpeg', res2)
    # cv2.waitKey(0)
    print(center)
    max_num = max(center[0][0], max(center[1][0], center[2][0]))
    min_num = min(center[0][0], min(center[1][0], center[2][0]))
    print(max_num)
    print(min_num)
    for i in range(K):
        if center[i][0] != max_num and center[i][0] != min_num:
            mid_num = center[i][0]
    ret, binary_0 = cv2.threshold(res2, min_num + 1, 255, cv2.THRESH_BINARY)
    print(binary_0)
    cv2.imshow("threshold after s_channel image", binary_0)
    cv2.imwrite('threshold after s_channel image.jpeg', binary_0)
    # cv2.waitKey(0)

    '''frame_0 = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h_channel = frame_0[:, :, 0]
    cv2.imshow("one image", h_channel)
    cv2.waitKey(0)
    s_channel = frame_0[:, :, 1]
    cv2.imshow("one image", s_channel)
    cv2.waitKey(0)
    v_channel = frame_0[:, :, 2]
    cv2.imshow("one image", v_channel)
    cv2.waitKey(0)
    frame_1 = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            if s_channel[i][j] >= 35 and ((h_channel[i][j] >= 0 and h_channel[i][j] < 24) or h_channel[i][j] >= 140):
                frame_1[i][j] = 255
    cv2.imshow("one image", frame_1)
    cv2.waitKey(0)'''

    frame_0 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray image", frame_0)
    cv2.imwrite('gray image.jpeg', frame_0)
    # cv2.waitKey(0)

    edges = cv2.Sobel(frame_0, cv2.CV_16S, 1, 1)
    edges = cv2.convertScaleAbs(edges)
    cv2.imshow("sobel image", edges)
    cv2.imwrite('sobel image.jpeg', edges)
    # cv2.waitKey(0)

    edgesh = cv2.Sobel(frame_0, cv2.CV_16S, 1, 0)
    edgesh = cv2.convertScaleAbs(edgesh)
    cv2.imshow("sobel horizontal image", edgesh)
    cv2.imwrite('sobel horizontal image.jpeg', edgesh)
    # cv2.waitKey(0)

    edgesv = cv2.Sobel(frame_0, cv2.CV_16S, 0, 1)
    edgesv = cv2.convertScaleAbs(edgesv)
    cv2.imshow("sobel vertical image", edgesv)
    cv2.imwrite('sobel vertical image.jpeg', edgesv)
    # cv2.waitKey(0)

    grad = cv2.addWeighted(edgesh, 0.5, edgesv, 0.5, 0)
    cv2.imshow("vertical + horizontal image", grad)
    cv2.imwrite('vertical + horizontal image.jpeg', grad)
    # cv2.waitKey(0)

    gradd = cv2.addWeighted(grad, 0.7, edges, 0.3, 0)
    cv2.imshow("sobel + vertical + horizontal image", gradd)
    cv2.imwrite('sobel + vertical + horizontal image.jpeg', gradd)
    # cv2.waitKey(0)

    # kernel_size = 9
    kernel_size_temp = round(((height * width) / (80 * 80)) ** 0.5)
    if kernel_size_temp % 2 == 0:
        kernel_size_temp = kernel_size_temp + 1
    kernel_size = max(9, kernel_size_temp)
    print(kernel_size)
    frame_1 = cv2.GaussianBlur(gradd, (kernel_size, kernel_size), 1)
    cv2.imshow("GaussianBlur image", frame_1)
    cv2.imwrite('GaussianBlur image.jpeg', frame_1)
    # cv2.waitKey(0)

    # kernel_size = 5
    kernel_size_temp = round(((height * width) / (180 * 180)) ** 0.5)
    if kernel_size_temp % 2 == 0:
        kernel_size_temp = kernel_size_temp + 1
    kernel_size = max(5, kernel_size_temp)
    print(kernel_size)
    frame_2 = cv2.medianBlur(frame_1, kernel_size)
    cv2.imshow("medianBlur image", frame_2)
    cv2.imwrite('medianBlur image.jpeg', frame_2)
    # cv2.waitKey(0)
    print('frame_2', frame_2)

    # ret, binary_1 = cv2.threshold(frame_2, 100, 255, cv2.THRESH_BINARY)
    low_bound = 100
    upper_bound = 150
    frame_3 = cv2.Canny(frame_2, low_bound, upper_bound) # + cv2.Canny(s_channel, low_bound, upper_bound)
    cv2.imshow("Canny image", frame_3)
    cv2.imwrite('Canny image.jpeg', frame_3)
    # cv2.waitKey(0)

    lines_0 = cv2.HoughLinesP(frame_3, rho=1, theta=np.pi/180, threshold=round((height + width) / 20),
                              minLineLength=round((height + width) / 20), maxLineGap=round((height + width) / 20))
    lines = []
    if lines_0 is not None:
        lines.extend(lines_0)
    #if lines_1 is not None:
    #    lines.extend(lines_1)
    print(type(lines_0))
    frame_blank = np.zeros((height,width,3), np.uint8)
    if lines is not []:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame_blank, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)
    cv2.imshow("HoughLines image", frame_blank)
    cv2.imwrite('HoughLines image.jpeg', frame_blank)

    '''frame_blank = np.zeros((height, width, 3), np.uint8)
        for i in range(width):
            for j in range(height):
                if frame_2[j][i] == 0:
                    frame_blank[j][i][0] = 0
                    frame_blank[j][i][1] = 0
                    frame_blank[j][i][2] = 0
                elif frame_2[j][i] == 1:
                    frame_blank[j][i][0] = 255
                    frame_blank[j][i][1] = 255
                    frame_blank[j][i][2] = 255'''

    gray = cv2.cvtColor(frame_blank, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow('Gaussian', gray)
    gray = np.float32(gray)
    # harris detection
    dst = cv2.cornerHarris(gray, 2, 3, 0.02)
    # dst = cv2.dilate(dst, None)
    # dst = cv2.erode(dst, None)
    # threshold
    frame_blank[dst > 0.0001 * dst.max()] = [225, 0, 0]
    cv2.imshow('Harris', frame_blank)
    cv2.imwrite('Harris.jpeg', frame_blank)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    cv2.imshow('mask', mask)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    img2_fg = cv2.bitwise_and(frame_blank, frame_blank, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    cv2.imshow('mask_result', dst)
    cv2.imwrite('mask_result.jpeg', dst)

    '''frame_blank_1 = np.zeros((height, width, 3), np.uint8)
    for i in range(0, width):
        for j in range(0, height):
            if dst[j][i][0] == 255 and dst[j][i][1] == 0 and dst[j][i][2] == 0:
                frame_blank_1[j][i] = [255, 255, 255]
    cv2.imshow('binary_result', frame_blank_1)'''

    shape = dst.shape
    height = shape[0]
    width = shape[1]
    print(height, width)
    leftup_min = float("inf")
    rightup_min = float("inf")
    leftdown_min = float("inf")
    rightdown_min = float("inf")
    leftup_coor = [0, 0]
    rightup_coor = [width - 1, 0]
    leftdown_coor = [0, height - 1]
    rightdown_coor = [width - 1, height - 1]

    # Calculate four corner location
    for i in range(0, width):
        for j in range(0, height):
            if dst[j][i][0] == 225 and dst[j][i][1] == 0 and dst[j][i][2] == 0:
                distance_leftup = math.pow((j - 0), 2) + math.pow((i - 0), 2)
                distance_leftup = math.sqrt(distance_leftup)
                if distance_leftup < leftup_min:
                    leftup_min = distance_leftup
                    leftup_coor = [i, j]
                distance_rightup = math.pow((j - 0), 2) + math.pow((i - width), 2)
                distance_rightup = math.sqrt(distance_rightup)
                if distance_rightup < rightup_min:
                    rightup_min = distance_rightup
                    rightup_coor = [i, j]
                distance_leftdown = math.pow((j - height), 2) + math.pow((i - 0), 2)
                distance_leftdown = math.sqrt(distance_leftdown)
                if distance_leftdown < leftdown_min:
                    leftdown_min = distance_leftdown
                    leftdown_coor = [i, j]
                distance_rightdown = math.pow((j - height), 2) + math.pow((i - width), 2)
                distance_rightdown = math.sqrt(distance_rightdown)
                if distance_rightdown < rightdown_min:
                    rightdown_min = distance_rightdown
                    rightdown_coor = [i, j]

    cv2.circle(dst, leftup_coor, 9, (0, 0, 255), thickness=8)
    cv2.circle(dst, rightup_coor, 9, (0, 0, 255), thickness=8)
    cv2.circle(dst, leftdown_coor, 9, (0, 0, 255), thickness=8)
    cv2.circle(dst, rightdown_coor, 9, (0, 0, 255), thickness=8)
    cv2.imshow('corners result', dst)
    cv2.imwrite('corners result.jpeg', dst)
    '''
    A------------------------------------B
    .                                    .
    .                                    .
    .                   .---------.      .
    .                   .---------.      .
    C------------------------------------D
    '''
    # detect the angle of the check
    distance_AB = math.pow((leftup_coor[0] - rightup_coor[0]), 2) + math.pow((leftup_coor[1] - rightup_coor[1]), 2)
    distance_AB = math.sqrt(distance_AB)
    distance_CD = math.pow((leftdown_coor[0] - rightdown_coor[0]), 2) + math.pow((leftdown_coor[1] - rightdown_coor[1]), 2)
    distance_CD = math.sqrt(distance_CD)

    distance_AC = math.pow((leftup_coor[0] - leftdown_coor[0]), 2) + math.pow((leftup_coor[1] - leftdown_coor[1]), 2)
    distance_AC = math.sqrt(distance_AC)
    distance_BD = math.pow((rightup_coor[0] - rightdown_coor[0]), 2) + math.pow((rightup_coor[1] - rightdown_coor[1]), 2)
    distance_BD = math.sqrt(distance_BD)

    if (distance_AB + distance_CD) > (distance_AC + distance_BD):
        result_width = 1000
        result_height = 500
    else:
        result_width = 500
        result_height = 1000
    src_points = np.array([leftup_coor, rightup_coor, leftdown_coor, rightdown_coor], dtype="float32")
    dst_points = np.array([[0., 0.], [result_width, 0.], [0., result_height], [result_width, result_height]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    perspective = cv2.warpPerspective(frame, M, (result_width, result_height), cv2.INTER_LINEAR)

    perspective = cv2.cvtColor(perspective, cv2.COLOR_RGB2BGR)
    cv2.imshow("output image", perspective)
    cv2.imwrite('output image.jpeg', perspective)

    result = 1
    return result

########################################################################################################################

def runon_image(path) :
    frame = cv2.imread(path)
    height, width, channels = frame.shape
    print(frame.shape)
    if height > 1000 or width > 1000:
        factor = max(height, width)
        frame = cv2.resize(frame, None, fx=1000/factor, fy=1000/factor, interpolation=cv2.INTER_LINEAR)
        height, width, channels = frame.shape
    cv2.imshow("original image", frame)
    # cv2.waitKey(0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame.shape)
    cv2.imshow("RGB image", frame)
    cv2.imwrite('RGB image.jpeg', frame)
    # cv2.waitKey(0)

    detections_in_frame = detect_corner(frame, height, width)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("output image", frame)
    # cv2.imwrite('output image.jpeg', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detections_in_frame

def runon_folder(path) :
    files = None
    if(path[-1] != "/"):
        path = path + "/"
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    all_detections = 0
    print(files)
    for f in files:
        print(f)
        if f != 'samples/.DS_Store':
            f_detections = runon_image(f)
            all_detections += f_detections
    return all_detections

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', default='samples', help="requires path")
    args = parser.parse_args()
    folder = args.folder
    if folder is None :
        print("Folder path must be given \n Example: python proj1.py -folder images")
        sys.exit()
    if folder is not None :
        all_detections = runon_folder(folder)
        print("total of ", all_detections, " detections")
    cv2.destroyAllWindows()