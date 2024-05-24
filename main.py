import cv2
import numpy as np

cam = cv2.VideoCapture('Lane_Detection_Test_Video_01.mp4')

while True:

    ret, frame = cam.read()

    if ret is False:
        break

    new_width = 356
    new_height = 200

# 2 - Resize the original frame
    frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow('- 2 - Original video - Resized', frame)

# 3 - Turn the resized frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('- 3 - Grayscale resized frame', grayscale_frame)

# 4 - Selecting only the road using a trapezoid
    trapezoid_frame = np.zeros((new_height, new_width), dtype=np.uint8)
    # create the 4 points of the trapezoid
    upper_left = (int(new_width * 0.45), int(new_height * 0.76))
    upper_right = (int(new_width * 0.55), int(new_height * 0.77))
    lower_left = (0, new_height)
    lower_right = (new_width, new_height)

    # place the points intro an array in trigonometric order
    trapezoid_points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    # place the trapezoid into the black frame
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_points, 1)

    # showing the black frame with the white trapezoid
    cv2.imshow('- 4 - Black Frame with Trapezoid', trapezoid_frame * 255)

    # create a frame containing only the grayscale road
    trapezoid_grayscale_road = trapezoid_frame * grayscale_frame
    cv2.imshow('- 4 - Grayscale Road', trapezoid_grayscale_road)

# 5 - Top-Down View
    # create an array with the points of the corners of the screen
    screen_corners_points = np.array([(new_width, 0), (0, 0), (0, new_height), (new_width, new_height)], dtype=np.float32)

    # convert the trapezoid points array from int to float
    trapezoid_points = np.float32(trapezoid_points)

    # create the "magical matrix"
    magical_matrix = cv2.getPerspectiveTransform(trapezoid_points, screen_corners_points)

    # create the frame with the Top-Down view of the road
    top_down_view_road = cv2.warpPerspective(trapezoid_grayscale_road, magical_matrix, (new_width, new_height))
    cv2.imshow("- 6 - Top-Down View", top_down_view_road)

# 6 - Adding Blur
    # create the blur area square matrix
    ksize = (5, 5)
    blurred_td_view_road = cv2.blur(top_down_view_road, ksize)
    cv2.imshow("- 6 - Blurred TD View", blurred_td_view_road)

# 7 - Edge detection
    # create the Sobel matrices:
    sobel_vertical_matrix = np.float32([[-1, -2, -1],
                                        [0, 0, 0],
                                        [1, 2, 1]])
    sobel_horizontal_matrix = np.transpose(sobel_vertical_matrix)

    # create the two frames with the filters applied
    sobel_vertical_frame = cv2.filter2D(blurred_td_view_road, -1, sobel_vertical_matrix)
    sobel_horizontal_frame = cv2.filter2D(blurred_td_view_road, -1, sobel_horizontal_matrix)

    # display the two Sobel Filter frames
    cv2.imshow("- 7 - Sobel Vertical Filter", sobel_vertical_frame)
    cv2.imshow("- 7 - Sobel Horizontal Filter", sobel_horizontal_frame)

    # convert the matrices to float32
    sobel_vertical_frame = np.float32(sobel_vertical_frame)
    sobel_horizontal_frame = np.float32(sobel_horizontal_frame)

    # combine the two matrices
    road_lines = np.sqrt(np.square(sobel_vertical_frame) + np.square(sobel_horizontal_frame))

    # convert the matrix to uint8
    road_lines = cv2.convertScaleAbs(road_lines)
    cv2.imshow("- 7 - Road Lines", road_lines)

# 8 - Binarize the frame
    threshhold = 60
    thr, binary_road_lines = cv2.threshold(road_lines, threshhold, 255, cv2.THRESH_BINARY)
    cv2.imshow("- 8 - Binary Road Lines", binary_road_lines)

# 9 - Road Lines Coordinates
    # create a copy of the frame
    clean_road_lines = binary_road_lines.copy()

    # set the left 5%, right 20% and the bottom 4% of pixels to black for cleaning noise
    clean_road_lines[0:new_height, 0:int(new_width * 0.05)] = 0  # left 5%
    clean_road_lines[0:new_height, int(new_width * 0.80):new_width] = 0  # right 20%
    clean_road_lines[int(new_height * 0.96):new_height, 0:new_width] = 0  # bottom 4%

    cv2.imshow("- 9 - Clean Road Lines", clean_road_lines)

    # splitting into 2 frames
    frame_half = int(new_width * 0.5)
    left_road_lines = clean_road_lines[:, :frame_half]
    right_road_lines = clean_road_lines[:, frame_half:]

    cv2.imshow("- 9 - Left Road Lines", left_road_lines)
    cv2.imshow("- 9 - Right Road Lines", right_road_lines)

    # get the two arrays of white points
    # we will iterate from right to left because np.argwhere returns points as (y, x)
    # and we need them to be (x, y)
    left_white_points = np.argwhere(left_road_lines > 0)[:, ::-1]
    right_white_points = np.argwhere(right_road_lines > 0)[:, ::-1]

    # for the right white points we add the new_width//2 value to x coordinate
    right_white_points[:, 0] += new_width // 2

    # get the 4 sets of coordinates
    left_xs = left_white_points[:, 0]
    left_ys = left_white_points[:, 1]
    right_xs = right_white_points[:, 0]
    right_ys = right_white_points[:, 1]

# 10 - Get the lines that detect edges of the lanes
    # get the a and b for left and right sets of points
    if np.any(left_xs) and np.any(left_ys):
        b_left, a_left = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    if np.any(right_xs) and np.any(right_ys):
        b_right, a_right = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    # get the x from y = ax + b ==> x = (y - b) / a
    top_y = 0
    bottom_y = new_height
    x_limit = 10 ** 8
    # for the left line
    left_top_x = int(np.abs((top_y - b_left) / a_left))
    if left_top_x <= x_limit:
        left_top = left_top_x, top_y

    left_bottom_x = int(np.abs((bottom_y - b_left) / a_left))
    if left_bottom_x <= x_limit:
        left_bottom = left_bottom_x, bottom_y

    # for the right line
    right_top_x = int(np.abs((top_y - b_right) / a_right))
    if right_top_x <= x_limit:
        right_top = right_top_x, top_y

    right_bottom_x = int(np.abs((bottom_y - b_right) / a_right))
    if right_bottom_x <= x_limit:
        right_bottom = right_bottom_x, bottom_y

    # drawing the lines
    cv2.line(clean_road_lines, left_top, left_bottom, (100, 0, 0), 4)
    cv2.line(clean_road_lines, right_top, right_bottom, (200, 0, 0), 4)
    cv2.line(clean_road_lines, (new_width // 2, 0), (new_width // 2, new_height), (255, 0, 0), 1)
    cv2.imshow("- 10 - Lines of Lane Edges", clean_road_lines)

# 11 - Functional Lane Detector
    # for the left line
    left_blank_frame = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.line(left_blank_frame, left_top, left_bottom, (255, 0, 0), 5)

    left_magical_matrix = cv2.getPerspectiveTransform(screen_corners_points, np.float32(trapezoid_points))
    left_line_frame = cv2.warpPerspective(left_blank_frame, left_magical_matrix, (new_width, new_height))

    left_frame_points = np.argwhere(left_line_frame > 0)[:, ::-1]
    left_xs = left_frame_points[:, 0]
    left_ys = left_frame_points[:, 1]

    # for the right line
    right_blank_frame = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.line(right_blank_frame, right_top, right_bottom, (255, 0, 0), 5)

    right_magical_matrix = cv2.getPerspectiveTransform(screen_corners_points, np.float32(trapezoid_points))
    right_line_frame = cv2.warpPerspective(right_blank_frame, right_magical_matrix, (new_width, new_height))

    right_frame_points = np.argwhere(right_line_frame > 0)[:, ::-1]
    right_xs = right_frame_points[:, 0]
    right_ys = right_frame_points[:, 1]

    # get a copy of the original frame
    lane_detector = frame.copy()
    lane_detector[left_ys, left_xs] = (150, 0, 150)  # red left line
    lane_detector[right_ys, right_xs] = (50, 250, 50)  # green right line

    cv2.imshow("- 10 - Lane Detector", lane_detector)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()