import cv2
from realsense_depth import *
from math import cos, sin, sqrt, asin, degrees


required_length = 300 # необходимая дистанция от ряда в миллиметрах

cam_angle = [0.7,0.61,0.52,0.44]
x, y = 321, 240
point1 = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()
rows = 'both'
one_rad = 640/1.57 #узнаём сколько пикселей вмещается в одну радиану. Угол обзора камеры 90'

while True:
    angles = []
    ride_commands = []
    # получаем кадр
    ret, depth_frame, color_frame = dc.get_frame()
    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

    for cam_rad in cam_angle:
        
        # находим нужные опорные точки
        lengh_px = cam_rad*one_rad #длина кадра в пикс
        point_left = (int((320-lengh_px)), y)
        point_right = (int((320+lengh_px)), y)

        # опредедляем левую точку маршрута
        distance_left = depth_frame[point_left[1], point_left[0]]
        if distance_left <= 0: distance_left = 0.1
        lengh_mm_left = distance_left * sin(cam_rad)
        ride_commands.append(lengh_mm_left)
        if lengh_mm_left <= 0: lengh_mm_left = 0.1
        line_pos_left = int(((lengh_mm_left-required_length)*lengh_px)//lengh_mm_left)
        # опредедляем правую точку маршрута
        distance_right = depth_frame[point_right[1], point_right[0]]
        if distance_right <= 0: distance_right = 0.1
        lengh_mm_right = distance_right * sin(cam_rad)
        ride_commands.append(lengh_mm_right)
        if lengh_mm_right <= 0: lengh_mm_right = 0.1
        line_pos_right = int(((lengh_mm_right-required_length)*lengh_px)//lengh_mm_right)

        # Изображаем точки на кадре
        cv2.circle(depth_image, (int(point_left[0]), int(point_left[1])), 5, (0, 255, 0), -1)
        cv2.circle(depth_image, (int(point_right[0]), int(point_right[1])), 5, (0, 255, 0), -1)

        cv2.putText(depth_image, "{}".format(distance_left), (point_left[0]-5, point_left[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(depth_image, "{}".format(distance_right), (point_right[0]-5, point_right[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.circle(depth_image, (x -line_pos_left, int(point_left[1] +(cam_rad*200))), 4, (255, 51, 255), 2)
        cv2.circle(depth_image, (x +line_pos_right, int(point_right[1] +(cam_rad*200))), 4, (51, 255, 255), 2)

        cv2.putText(depth_image, "{}mm".format(int(lengh_mm_left)), (x-line_pos_left + 10, int(point_left[1] +(cam_rad*200))), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(depth_image, "{}mm".format(int(lengh_mm_right)), (x+line_pos_right + 10, int(point_right[1] +(cam_rad*200))), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2) 

        # Находим потенциальный угол отклонения робота от заданного маршрута
        try:
            angle_left = degrees(abs(asin(required_length/distance_left)-cam_rad))
            angles.append(angle_left)
        except: angles.append(45)
        try:
            angle_right = degrees(abs(asin(required_length/distance_right)-cam_rad))
            angles.append(angle_right)
        except: angles.append(45)
    deflection_angle_left = int((angles[0] + angles[2] +angles[4] + angles[6]) / 4)  
    deflection_angle_right = int((angles[1] + angles[3] +angles[5] + angles[7]) / 4)

    # Рисуем точку центра кадра
    distance1 = depth_frame[point1[1], point1[0]]
    cv2.circle(depth_image, (int(point1[0]), int(point1[1])), 5, (0, 255, 0), -1)
    cv2.putText(depth_image, "{}mm".format(distance1), (point1[0], point1[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    
    # Выбираем, показатели какой из сторон релевантны для работы
    if all(0 <= command <= (required_length * 2) for command in ride_commands):
        rows = 'both'
        cv2.putText(depth_image, "both", (440, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif all(0 <= command <= (required_length * 2) for command in ride_commands[1::2]):
        rows = 'right'
        cv2.putText(depth_image, "right", (440, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif all(0 <= command <= (required_length * 2) for command in ride_commands[::2]):
        rows = 'left'
        cv2.putText(depth_image, "left", (440, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
     
    # Принимаем решение о корректировке движения
    if rows == 'both':

        if sum(required_length-30 <= command <= required_length+30 for command in ride_commands) >= len(ride_commands) - 4: 
            cv2.putText(depth_image, "Pryamo", (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if (ride_commands[0] <= required_length-30) and (ride_commands[1] >= required_length+30) and (ride_commands[2] <= required_length-30) and (ride_commands[3] >= required_length+30) and (ride_commands[4] <= required_length-30) and (ride_commands[5] >= required_length+30) and (ride_commands[6] <= required_length-30) and (ride_commands[7] >= required_length+30): 
            cv2.putText(depth_image, "Napravo " + str(deflection_angle_left) + ' deg.', (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if (ride_commands[0] >= required_length+30) and (ride_commands[1] <= required_length-30) and (ride_commands[2] >= required_length+30) and (ride_commands[3] <= required_length-30) and (ride_commands[4] >= required_length+30) and (ride_commands[5] <= required_length-30) and (ride_commands[6] >= required_length+30) and (ride_commands[7] <= required_length-30): 
            cv2.putText(depth_image, "Nalevo " + str(deflection_angle_right) + ' deg.', (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if rows == 'left':

        if sum(required_length-30 <= command <= required_length+30 for command in ride_commands[::2]) >= len(ride_commands[::2]) - 1: 
            cv2.putText(depth_image, "Pryamo", (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if (ride_commands[0] <= required_length-30) and (ride_commands[2] <= required_length-30) and (ride_commands[4] <= required_length-30) and (ride_commands[6] <= required_length-30): 
            cv2.putText(depth_image, "Napravo " + str(deflection_angle_left) + ' deg.', (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if (ride_commands[0] >= required_length+30) and (ride_commands[2] >= required_length+30) and (ride_commands[4] >= required_length+30) and (ride_commands[6] >= required_length+30): 
            
            cv2.putText(depth_image, "Nalevo " + str(deflection_angle_right) + ' deg.', (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if rows == 'right':

        if sum(required_length-30 <= command <= required_length+30 for command in ride_commands[1::2]) >= len(ride_commands[1::2]) - 1: 
            cv2.putText(depth_image, "Pryamo", (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
        if (ride_commands[1] >= required_length+30) and (ride_commands[3] >= required_length+30) and (ride_commands[5] >= required_length+30) and (ride_commands[7] >= required_length+30): 
            
            cv2.putText(depth_image, "Napravo " + str(deflection_angle_left) + ' deg.', (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if (ride_commands[1] <= required_length-30) and (ride_commands[3] <= required_length-30) and (ride_commands[5] <= required_length-30) and (ride_commands[7] <= required_length-30): 
            
            cv2.putText(depth_image, "Nalevo " + str(deflection_angle_right) + ' deg.', (20, 20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Визуализируем кадр
    cv2.imshow("depth frame", depth_image)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
dc.release()
cv2.destroyAllWindows()

