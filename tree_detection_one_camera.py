#https://pysource.com/2018/05/14/optical-flow-with-lucas-kanade-method-opencv-3-4-with-python-3-tutorial-31/

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('model/yolov8n_tree_10ep3/weights/best.pt')

cap = cv2.VideoCapture("Y:/OpenCV/vid/000_vid_cut.mp4")
# cap = cv2.VideoCapture(0)


# Создаём первый фрейм для настройки функций
num = 0 # кол-во деревьев
dist = 112 # пройденная дистанция в пикс
box_0, box_1, box_2 = [], [], []
tree_pixel, prev_tree_pixel = 0, 0
center_pixel = 0
_, frame = cap.read()

frame = cv2.resize(frame, (224, 224))
frame = cv2.flip(frame, 1)
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize = (15, 15), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def select_point(count_dist,  x = 224, y = 200 ):
    global dist, point, old_points
    dist += count_dist
    point = (x, y)
    old_points = np.array([[224, 200]], dtype=np.float32)


point = ()
old_points = np.array([[]])
select_point(0)

tree_data = []

while True: 
    _, frame = cap.read()
    if frame is None:
        dist += abs(point[0] - coord[0])
        break
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
    old_gray = gray_frame.copy()
    old_points = new_points
    coord = old_points[0]
    
    x, y = new_points.ravel()
    # print(old_points)
    # print(gray_frame)
    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 255), -1)


    results = model.predict(source=frame, show= True, conf =0.65)

    boxes = results[0].boxes.xyxy.tolist()
    conf = results[0].boxes.conf.tolist()
    # if boxes !=[] and conf[0] > 0.79:
    if boxes !=[]:
        box_3 = box_2
        box_2 = box_1
        box_1 = box_0
        box_0 = boxes[0]

        center_tree = ((int(box_0[2])-int(box_0[0]))//2)+int(box_0[0])
        right_side_tree = int(box_0[2])
        if center_tree < 120 or num ==0:
            tree_pixel = dist + center_tree - coord[0]
        else:    
            tree_pixel = dist + right_side_tree - coord[0]
        
    else:
        box_3 = box_2
        box_2 = box_1
        box_1 = box_0
        box_0 = []
        


   
    print(coord[1])
    if coord[0] <= 1 or coord[0] >= 224 or coord[1] <= 1 or coord[1] >= 224 :
        
        select_point(abs(point[0] - coord[0]))


    if (box_3==[] and box_2==[] and box_1==[] and box_0 != []) or (box_1==[] and box_0 != [] and abs(tree_pixel - prev_tree_pixel) > 210):
        
        
        
        cv2.circle(frame, (center_tree, 100), 20, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        
        if  center_tree > 74 and (abs(tree_pixel - prev_tree_pixel) > 210 or num==0):
            print(prev_tree_pixel)
            print(tree_pixel)
            num+=1
            tree_data.append( (tree_pixel, 'Semerenko', num))
            
            prev_tree_pixel = tree_pixel
            center_pixel = coord[0]
            
            
        else:
            print(prev_tree_pixel)
            print(tree_pixel)
            print('stop')
            
        
        
    frame = cv2.putText(frame, str(num), (10, 160), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 4, color = (255, 255, 0), thickness= 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
print(dist)
print(tree_data)
cv2.destroyAllWindows()