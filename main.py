# Face Detection Based Attendance System (Python Based Project)
# Shashwat Rauthan
# Btech CSE Vth sem
# Sec. B (54)


# Importing Libraries
import cv2
import face_recognition
import numpy as np
import os
import time
from datetime import datetime



# this function finds encodings for each image in images list
def encode_image(images):
    print('Building Student\'s Images Encoding List')
    encodings_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  #changing image from BGR to RGB
        face_pos = face_recognition.face_locations(img_rgb)[0]        #finding face locations
        y1, x2, y2, x1 = face_pos
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)           #drawing rectangle over students images
        cv2.rectangle(img, (x1, y2 + 35), (x2, y2), (0, 0, 255), cv2.FILLED)    #rectangle for text
        text= "Face Detected"
        cv2.putText(img, text, (x1, y2 + 30), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 2)
        cv2.namedWindow('Students Images', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Students Images', 600, 600)
        cv2.imshow('Students Images', img)                #showing students images being encoded
        cv2.waitKey(1)

        encode = face_recognition.face_encodings(img_rgb)[0]             #finding encodings of each image
        encodings_list.append(encode)

    cv2.destroyWindow("Students Images")
    return encodings_list


# this function marks attendance in csv file with current time stamp
def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        # building a list of students already marked present, to prevent multiple entries for same student
        csv_list = f.readlines()
        names_list = []
        for line in csv_list:
            entry = line.split(',')[0]
            names_list.append(entry)

        # marking attendance for only those students who are not already marked
        if name not in names_list:
            curr_time = datetime.now()
            time_stamp = curr_time.strftime('%I:%M:%S %p')
            f.writelines(f'\n{name},{time_stamp},Present')




path = 'Students_Images'      #Path of the folder where Images of all the Students are stored
images = []
class_names = []
my_list = os.listdir(path)
print("Images found for these Students:")
print(my_list)

# storing images in images list & names in class_names list
for cloc in my_list:
    current_img = cv2.imread(f'{path}/{cloc}')
    images.append(current_img)
    name = os.path.splitext(cloc)[0]
    class_names.append(name)



# finding encodings for each image in images list
students_encodings_list = encode_image(images)
print('Student\'s Images Encoding List Built Successfully')

# initiating webcam capture
print('Starting Webcam')
cap = cv2.VideoCapture(0)



timeout = time.time() + 10                   #setting timeout time 10 seconds
attended_flag = 0

while time.time() < timeout:                   #timeout at 10 sec
    # reading video frame by frame & processing each frame for face match
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              #changing image from BGR to RGB

    # finding faces in frame, because frame can have multiple faces
    face_pos_curr_frame = face_recognition.face_locations(img_rgb)
    encode_curr_frame = face_recognition.face_encodings(img_rgb, face_pos_curr_frame)


    # finding matches
    for face_encoding, face_loc in zip(encode_curr_frame, face_pos_curr_frame):
        matches = face_recognition.compare_faces(students_encodings_list, face_encoding)
        face_distance = face_recognition.face_distance(students_encodings_list, face_encoding)
        print(face_distance)

        match_index = np.argmin(face_distance)       # finding minimum distance, lower the distance better is the match
        print('MatchIndex :', match_index)

        # creating markings over live feed frames & marking attendance
        if matches[match_index]:
            name = class_names[match_index]
            print(name)
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)         #rectangle for face
            cv2.rectangle(img, (x1, y1 - 25), (x2-50, y1), (255, 255, 0), cv2.FILLED)  # rectangle for "Match" (top)
            cv2.rectangle(img, (x1, y2), (x2, y2 + 30), (255, 255, 0), cv2.FILLED)  # rectangle for Name  (bottom)
            text = "Match:"
            cv2.putText(img, text, (x1 , y1), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 2)        #Match text
            cv2.putText(img, name, (x1 , y2 + 25), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 2)        #Name text


            # marking attendance
            mark_attendance(name)
            print("Attendance Marked")
            attended_flag = 1


    cv2.imshow('Webcam', img)

    #press 'Esc' to close the program
    if cv2.waitKey(1) == 27:
        break

if attended_flag == 0:
    print("No Match Found")

#releasing camera
cap.release()
cv2.destroyAllWindows()
