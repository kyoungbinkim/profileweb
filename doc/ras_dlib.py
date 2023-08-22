from picamera.array import PiRGBArray
import time
import cv2
import dlib
import picamera
import numpy as np
from imutils import face_utils
import glob
import tkinter as tk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

## 얼굴번호 list 생성
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

koomin = []

## face detector, predictor 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

## 바탕화면 이미지에 overlay하는 함수
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
     # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
    print(img2_fg)
    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


def face_remap(shape):
    remapped_image = shape.copy()
    # left eye brow
    remapped_image[17] = shape[26]
    remapped_image[18] = shape[25]
    remapped_image[19] = shape[24]
    remapped_image[20] = shape[23]
    remapped_image[21] = shape[22]
    # right eye brow
    remapped_image[22] = shape[21]
    remapped_image[23] = shape[20]
    remapped_image[24] = shape[19]
    remapped_image[25] = shape[18]
    remapped_image[26] = shape[17]
    # neatening
    remapped_image[27] = shape[0]

    return remapped_image

## tkinter 창에서 엔터가 입력되었을때 실행되는 event call
def send_email(event):
    # 엔터키로 입력받기
    email_addr = str(entry.get())
    print(email_addr)
    label = tk.Label(root, text = 'sneding email')
    label.pack()
    s = smtplib.SMTP('smtp.gmail.com',587)
    s.starttls()
    s.login('kmufacefilter@gmail.com','codxdirdupcupfvf') # 로그인
   
    msg = MIMEMultipart()
    msg['Subject'] = 'KMU face filter'      # 제목설정
    msg.attach(MIMEText('hi','plain'))      # 내용

    ## 첨부파일 1
    attachment = open('project/saved/s0.jpg','rb')      
    part = MIMEBase('application','octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('content-Disposition','attachment; filename= 1.jpg')
    msg.attach(part)

    ## 첨부파일 2
    attachment = open('project/saved/s1.jpg','rb')      
    part = MIMEBase('application','octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('content-Disposition','attachment; filename= 2.jpg')
    msg.attach(part)

    ## 첨부파일 3
    attachment = open('project/saved/s2.jpg','rb')      
    part = MIMEBase('application','octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('content-Disposition','attachment; filename= 3.jpg')
    msg.attach(part)

    ## 첨부파일 4
    attachment = open('project/saved/s3.jpg','rb')      
    part = MIMEBase('application','octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('content-Disposition','attachment; filename= 4.jpg')
    msg.attach(part)

    ## 이메일 보내기
    s.sendmail('kmufacefilter@gmail.com',email_addr,msg.as_string())
    s.quit()
    label = tk.Label(root, text = 'successfully sned email')
    label.pack()
    # 창 닫기
    root.destroy()

if __name__ == '__main__':
    # Initialize camera
    camera = picamera.PiCamera()
    camera.resolution = (500,300)
    camera.framerate = 15
    rawCapture = PiRGBArray(camera,size=(500,300))
    time.sleep(0.1)
    camera.start_preview(fullscreen=True)
    camera.stop_preview()

    saved_number = int(0)
    saved_img =[]           ## 저장이미지 배열 

    ## 첫번째 대기화면 
    stnadby_explain_img= np.zeros((300,500,3), np.uint8 )
    cv2.putText(stnadby_explain_img ,' 1~6 -> select filter', (30,50),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
    cv2.putText(stnadby_explain_img, '  S  ->     save    ', (30, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(stnadby_explain_img, ' save 4 photos to send  ', (30, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(stnadby_explain_img, ' enter any key to start  ', (30, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('result',stnadby_explain_img)
    cv2.waitKey()

    ## 촬영후 대기화면
    svaing_explain_img= np.zeros((300,500,3), np.uint8 )
    cv2.putText(svaing_explain_img, ' Q  -> retake photo', (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(svaing_explain_img, ' S  ->     save    ', (30, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(svaing_explain_img, '   enter any key   ', (30, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    ## 쿠민이 이미지 읽어오기
    for img in glob.glob("project/koomin*.png"):
        ktmp = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        koomin.append(ktmp)
        print(ktmp.shape)
    koomin.append(cv2.imread("project/oominall.png",cv2.IMREAD_UNCHANGED))
    ## 배경화면 읽어오기
    img_backgnd1 = cv2.imread('project/background.jpg')
    img_backgnd2 = cv2.imread('project/background2.jpg')
    img_backgnd3 = cv2.imread('project/background3.jpg')
    img_backgnd4 = cv2.imread('project/background4.jpg')
    frame_number = int(0)
    saved_number= int(0)
    filter = int(0)
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        image = frame.array
       
        key = cv2.waitKey(1) & 0xFF
        image = cv2.resize(image,None, interpolation=cv2.INTER_AREA,fx = 0.95, fy=1.0)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detector = detector(img_gray, 1)
        print("The number of faces detected : {}".format(len(face_detector)))

        result = image.copy()
        if filter == 1:
            tmp_1 = img_backgnd1.copy()
        elif filter == 3:
            tmp_1 = img_backgnd2.copy()
        elif filter ==4:
            tmp_1 = img_backgnd3.copy()
        elif filter ==5 :
            tmp_1 = img_backgnd4.copy()
        for face in face_detector:
            #cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),(0, 0, 255), 3)
            landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기
            landmark_list = []

            for p in landmarks.parts():
                landmark_list.append([p.x, p.y])
                #cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)
            
            ## 필터번호에 따라 필터 실행
            if filter == 0:
                ratio = float((landmark_list[31][0]-landmark_list[3][0])/koomin[0].shape[1])
                tmp_0 = cv2.resize(koomin[int(frame_number/10)%6], None,interpolation=cv2.INTER_AREA,fx = ratio-0.01, fy=ratio-0.01)
                result = overlay_transparent(result, tmp_0,landmark_list[36][0],int((landmark_list[41][1]+landmark_list[48][1])/2))
            elif filter == 1 or filter == 3 or filter == 4 or filter == 5:
                tmp_1 = cv2.resize(tmp_1, (image.shape[1], image.shape[0]))
                shape = predictor(img_gray, face)
                shape = face_utils.shape_to_np(shape)
                remapped_shape = np.zeros_like(shape)
                feature_mask = np.zeros((image.shape[0], image.shape[1]))
                remapped_shape = face_remap(shape)
                cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
                feature_mask = feature_mask.astype(np.bool)
                tmp_1[feature_mask] = image[feature_mask]
                result = tmp_1.copy()
            elif filter == 2:
                x=landmark_list[14][0]-landmark_list[2][0]
                y=landmark_list[33][1]-landmark_list[28][1]
                print(x)
                print(y)
                tmp_0 = cv2.resize(koomin[6],(x,y),interpolation=cv2.INTER_AREA)
                result = overlay_transparent(result, tmp_0, int((landmark_list[1][0] + landmark_list[15][0]) / 2), int((landmark_list[28][1] + landmark_list[33][1]) / 2))
        cv2.imshow('result', result)

        key = cv2.waitKey(10) & 0xff

        ## s 입력하면 사진저장
        if key == ord('s'):
            cv2.imwrite('project/saved/s%d.jpg'%(saved_number), result)
            saved_img.append(result)
            saved_number +=1
            if saved_number == 4:
                saved_number = 0
                cv2.imshow('result',svaing_explain_img)
                cv2.waitKey(3000)
                # 4개의 사진 분할
                tmp4 = np.zeros((300, 500, 3),np.uint8)
                print(saved_img[3].shape[0])
                height = 150
                width = 250
                tmp4[0:height, 0 :width] = cv2.resize(saved_img[0], (250,150))
                tmp4[0:height, width:width*2] = cv2.resize(saved_img[1], (250,150))
                tmp4[height:height*2, 0:width] = cv2.resize(saved_img[2], (250,150))
                tmp4[height:height*2,width:width*2] = cv2.resize(saved_img[3], (250,150))
                cv2.imshow('result', tmp4)
                while True:
                    key = cv2.waitKey() & 0xff
                    if key == ord('q'):
                        cv2.imshow('result',stnadby_explain_img)
                        cv2.waitKey()
                        saved_img.clear()
                        break
                    elif key == ord('s'):
                        root = tk.Tk()                      # TK생성
                        root.title("email")                 # 제목설정
                        root.geometry("300x100+50+50")      # 크기설정

                        label = tk.Label(root, text = 'write your email')   # 텍스트 쓰기
                        label.pack()

                        entry = tk.Entry(root,width=33)     # 엔터누르면  이벤트 실행
                        entry.bind("<Return>",send_email)
                        entry.pack()

                        root.mainloop()
                        cv2.waitKey(100)
                        cv2.imshow('result',stnadby_explain_img)
                        cv2.waitKey()
                        saved_img.clear()
                        break
            else:
                cv2.waitKey(500)
        ## q누르면 실행 종료
        elif key ==ord('q'):
            break
        elif key == ord('n'):
            filter +=1
            if filter == 6:
                filter = 0
        ## 1~6 번 입력으로 필터 선택
        elif key == ord('6') or key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5'):
            filter =int(key)-49

       
        frame_number+=1

        rawCapture.truncate(0)
    cv2.destroyAllWindows()