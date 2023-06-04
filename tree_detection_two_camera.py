import cv2
import numpy as np
from ultralytics import YOLO

#Подключаем дообученную модель
model = YOLO('model/yolov8n_tree_10ep3/weights/best.pt')
# Указываем источники кадров
# cap = [cv2.VideoCapture(f"Y:/OpenCV/vid/000_vid_cut.mp4"), cv2.VideoCapture(f"Y:/OpenCV/vid/000_vid_cut1.mp4") ]
cap = [cv2.VideoCapture(0), cv2.VideoCapture(1)]

num, tree_pixel, prev_tree_pixel, center_pixel = [[0, 0] for _ in range(4)] # переменные для работы алгоритма
dist = [112,112] 
box = [[[],[],[],[]],[[],[],[],[]]]
old_points = [np.array([[]]),np.array([[]])]

# Создаём первый фрейм для настройки функций
old_gray = []
for c in cap:
    ret, frame = c.read()
    if not ret:
        raise ValueError("Unable to capture video")
    resized_frame = cv2.resize(frame, (224, 224))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    old_gray.append(gray_frame)

# Lucas kanade params
lk_params = dict(winSize = (15, 15), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def count_dist(count_dist, idx):
    global dist, old_points, x , y
    dist[idx] += count_dist
    print(dist)
    x[idx] = 224
    y[idx] = 200
    old_points[idx] = np.array([[x[idx], y[idx]]], dtype=np.float32)

x, y = [224]*2, [200]*2
count_dist(0, 0)
count_dist(0, 1)

tree_data = [[] for _ in range(2)]

while True:
    frame = [cap[i].read()[1] for i in range(2)]
    if frame[0] is None or frame[1] is None:
        break
    frame[0] = cv2.resize(frame[0], (224, 224))
    frame[1] = cv2.resize(frame[1], (224, 224))
    frame[1] = cv2.flip(frame[1], 1)
    gray_frame = [cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY) for i in range(2)]
    new_points, status, error = [0,0], [[],[]], [[],[]]
    new_points[0], status[0], error[0] = cv2.calcOpticalFlowPyrLK(old_gray[0], gray_frame[0], old_points[0], None, **lk_params)
    new_points[1], status[1], error[1] = cv2.calcOpticalFlowPyrLK(old_gray[1], gray_frame[1], old_points[1], None, **lk_params)
    old_gray = gray_frame.copy()
    old_points = new_points.copy()
    coord = [old_points[i][0] for i in range(2)]
    for i in range(2):
        
        xy = new_points[i].ravel()
        x[i] = xy[0]
        y[i] = xy[1]
        cv2.circle(frame[i], (int(x[i]), int(y[i])), 5, (255, 0, 255), -1)

        results = model.predict(source=frame[i], show=True, conf=0.65)

        boxes = results[0].boxes.xyxy.tolist()
        conf = results[0].boxes.conf.tolist()

        if boxes != []:
            box[i][3], box[i][2], box[i][1], box[i][0] = box[i][2], box[i][1], box[i][0], boxes[0]

            center_tree = ((int(box[i][0][2])-int(box[i][0][0]))//2)+int(box[i][0][0])
            right_side_tree = int(box[i][0][2])
            if center_tree < 120 or num[i] == 0:
                tree_pixel[i] = dist[i] + center_tree - coord[i][0]
            else:
                tree_pixel[i] = dist[i] + right_side_tree - coord[i][0]
        else:
            box[i][3], box[i][2], box[i][1], box[i][0] = box[i][2], box[i][1], box[i][0],[]
            
        if coord[i][0] <= 1 or coord[i][0] >= 224 or coord[i][1] <= 1 or coord[i][1] >= 224:
            count_dist(abs(224 - coord[i][0]), i)
        
        if (box[i][3]==[] and box[i][2]==[] and box[i][1]==[] and box[i][0] != []) or (box[i][1]==[] and box[i][0] != [] and abs(tree_pixel[i] - prev_tree_pixel[i]) > 210):
            cv2.circle(frame[i], (center_tree, 100), 20, (0, 255, 0), 2)
            cv2.imshow(f"Camera {i+1}", frame[i])

            if  center_tree > 74 and (abs(tree_pixel[i] - prev_tree_pixel[i]) > 210 or num[i]==0):
                print(prev_tree_pixel[i], tree_pixel[i],coord[i][0], dist[i])
                num[i] += 1
                tree_data[i].append( (tree_pixel[i], 'Semerenko', num[i]))
                prev_tree_pixel[i] = tree_pixel[i]
                center_pixel[i] = coord[i][0]
            else:
                print(prev_tree_pixel[i], tree_pixel[i], coord[i][0], dist[i], 'stop')

        frame[i] = cv2.putText(frame[i], str(num[i]), (10, 160), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(255, 255, 0), thickness=2)
        cv2.imshow(f"Camera {i+1}", frame[i])

    key = cv2.waitKey(1)
    if key == 27:
        break

for i in range(2):
    cap[i].release()
print(dist[0], tree_data[0])
print(dist[1], tree_data[1])
cv2.destroyAllWindows()
