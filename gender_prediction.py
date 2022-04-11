import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

# Khởi tạo nguồn cấp dữ liệu webcam
frame = cv2.VideoCapture(0)

# sử dụng thuật toán MTCNN
detector= MTCNN()

# setup các model
emotion_model = "./data/_mini_XCEPTION.106-0.65.hdf5"
ageProto="./data/age_deploy.prototxt" # Kiến trúc mô hình cho mô hình phát hiện tuổi
ageModel="./data/age_net.caffemodel" # Trọng lượng mô hình được đào tạo trước để phát hiện tuổi
genderProto="./data/gender_deploy.prototxt" # Kiến trúc mô hình cho mô hình phát hiện giới tính
genderModel="./data/gender_net.caffemodel" # Trọng số mô hình được đào tạo trước để phát hiện giới tính

# Mỗi Mô hình Caffe áp đặt hình dạng của hình ảnh đầu vào cũng cần xử lý trước hình ảnh 
# giống như phân số trung bình để loại bỏ ảnh hưởng của các thay đổi không đồng bộ
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# danh sách các cụm tuổi
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# danh sách các giới tính
genderList=['Male','Female']
# danh sách cảm xúc
Emotions = ["angry","disgust","scared","happy","sad","surprised","neutral"]

# Tải tệp xếp tầng để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
# Tải mô hình phân chia cảm xúc
emotion_classifier = load_model(emotion_model,compile=False)
# Tải mô hình dự đoán tuoi
ageNet=cv2.dnn.readNet(ageModel,ageProto)
# Tải mô hình dự đoán giới tính
genderNet=cv2.dnn.readNet(genderModel,genderProto)


def ageAndgender():
    # khởi tạo một vòng lặp while chạy vô thời hạn hoặc cho đến khi tất cả các khung được lặp qua. 
    # Mã kèm theo bên trong được thực thi cho mỗi khung hình đến từ webcam. 
    # Vòng lặp cũng có thể bị phá vỡ nếu người dùng nhấn phím ‘q’
    while True:
        # tìm nạp các khung hình liên tiếp từ nguồn cấp dữ liệu webcam
        ret, img = frame.read()
        default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = face_cascade.detectMultiScale(image=default_img, scaleFactor=1.3, minNeighbors=5)
        # Bắt đầu một vòng lặp for lặp qua tất cả các mặt được phát hiện (và sử dụng hàm ‘cv2 rectangle’ để đặt một hộp giới hạn xung quanh nó)
        for x, y, w, h in face:
            # trả về khuôn mặt đã cắt từ hình ảnh
            roi = default_img[y:y + h, x:x + w]
            # chuyển khung hình thành một đốm màu để sẵn sàng cho đầu vào
            blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # đặt hình ảnh làm đầu vào
            genderNet.setInput(blob)
            # so sánh với mạng lưới giới tính có sẵn
            genderPreds = genderNet.forward()
            #Liệt kê giới tính tương ứng và để nó trong dấu ()
            gender = genderList[genderPreds[0].argmax()]
            # đặt hình ảnh làm đầu vào
            ageNet.setInput(blob)
            # so sánh với mạng lưới tuổi có sẵn
            agePreds = ageNet.forward()
            # Liệt kê tuổi tương ứng và để nó trong dấu ()
            age = ageList[agePreds[0].argmax()]
            # vẽ hình chữ nhật trên khuôn mặt
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #   x1,y1 ------
            #   |          |
            #   |          |  <--------- vẽ ra khung nhận diện khuôn mặt
            #   |          |
            #   --------x2,y2
            # ghi ký tự lên khung
            cv2.putText(img, f"{gender}, {age} year", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

        cv2.imshow("Gender and Age Prediction", img)
        
        # đợi phím 'q' được nhấn và ngắt khỏi vòng lặp
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
                break
        cv2.destroyAllWindows()

def emotion():
    
    # khởi tạo một vòng lặp while chạy vô thời hạn hoặc cho đến khi tất cả các khung được lặp qua. 
    # Mã kèm theo bên trong được thực thi cho mỗi khung hình đến từ webcam. 
    # Vòng lặp cũng có thể bị phá vỡ nếu người dùng nhấn phím ‘q’
    while True:
        # tìm nạp các khung hình liên tiếp từ nguồn cấp dữ liệu webcam
        ret, img = frame.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
        #Bắt đầu một vòng lặp for lặp qua tất cả các mặt được phát hiện (và sử dụng hàm ‘cv2 rectangle’ để đặt một hộp giới hạn xung quanh nó)
        for x, y, w, h in face:
            # trả về khuôn mặt đã cắt từ hình ảnh
            roi = gray[y:y + h, x:x + w]
            # thay đổi kích thước về 48x48 px
            roi = cv2.resize(roi, (48, 48))
            # chuyển đổi các giá trị thành từ 0-1
            roi = roi.astype("float") / 255.0
            # chuyển hình ảnh sang mảng 
            roi = img_to_array(roi)
            # trèn 1 trục mới vào ảnh
            roi = np.expand_dims(roi, axis=0)
            # Xác định hàm dự đoán (), hàm này nhận bộ phân loại cảm xúc 
            # và hình ảnh đầu vào và trả về kết quả dự đoán đầu ra bởi mô hình
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = Emotions[preds.argmax()]
            # vẽ hình chữ nhật trên khuôn mặt
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #   x1,y1 ------
            #   |          |
            #   |          |  <--------- vẽ ra khung nhận diện khuôn mặt
            #   |          |
            #   --------x2,y2
            # ghi ký tự lên khung
            cv2.putText(img, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
            #cv2.face.drawFacemarks(img,)

        cv2.imshow("Gender and Age Prediction", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
                break
    # Đóng tất cả các cửa sổ (Nếu có cửa sổ đang mở và đang chạy).
    cv2.destroyAllWindows()

 
