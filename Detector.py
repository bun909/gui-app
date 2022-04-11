import cv2
from time import sleep
from PIL import Image 

def main_app(name):
        # Tải tệp xếp tầng để phát hiện khuôn mặt
        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Đọc tệp dữ liệu đã đào tạo
        recognizer.read(f"./data/classifiers/{name}_classifier.xml")
        # Khởi tạo nguồn cấp dữ liệu webcam 
        cap = cv2.VideoCapture(0)
        # tham số dự đoán = 0
        pred = 0
        # khởi tạo một vòng lặp while chạy vô thời hạn hoặc cho đến khi tất cả các khung được lặp qua. 
        # Mã kèm theo bên trong được thực thi cho mỗi khung hình đến từ webcam. 
        # Vòng lặp cũng có thể bị phá vỡ nếu người dùng nhấn phím ‘q’ 
        while True:
            # tìm nạp các khung hình liên tiếp từ nguồn cấp dữ liệu webcam
            ret, frame = cap.read()
            # chuyển đổi ảnh sang thang độ xám
            default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Tìm kiếm các khuôn mặt trong hình ảnh bằng cách sử dụng tệp xếp tầng đã tải
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            # Nếu các khuôn mặt được tìm thấy, nó sẽ trả về vị trí 
            # của các khuôn mặt được phát hiện dưới dạng Rect(x, y, w, h)
            #Bắt đầu một vòng lặp for lặp qua tất cả các mặt được phát hiện (và sử dụng hàm ‘cv2 rectangle’ để đặt một hộp giới hạn xung quanh nó)
            for (x,y,w,h) in faces:

                # trả về khuôn mặt đã cắt từ hình ảnh
                roi_gray = gray[y:y+h,x:x+w]

                id,confidence = recognizer.predict(roi_gray)
                # Tính toán mức độ tin cậy khi so sánh hình ảnh từ camera so với dữ liệu đã đào tạo
                confidence = 100 - int(confidence)
                # tham số dự đoán = 0
                pred = 0
                if confidence > 50: # độ tin cậy lớn hơn 50%
                        # đếm cộng bằng 1
                            pred += +1
                            text = name.upper() # chuyển đổi tất cả các ký tự của tên thường thành chữ hoa
                            font = cv2.FONT_HERSHEY_PLAIN #font chữ
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            #   x1,y1 ------
                            #   |          |
                            #   |          |  <--------- vẽ ra khung nhận diện khuôn mặt
                            #   |          |
                            #   --------x2,y2
                            # ghi tên người được nhận diện lên ảnh
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                else:       # độ tin cậy nhỏ hơn 50%
                            pred += -1 # đếm trừ 1
                            text = "UnknownFace"
                            font = cv2.FONT_HERSHEY_PLAIN
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

            cv2.imshow("image", frame)

            # đợi phím 'q' được nhấn và ngắt khỏi vòng lặp
            if cv2.waitKey(20) & 0xFF == ord('q'):
                print(pred)
                if pred > 0 : 
                    dim =(124,124) # setup kích thước mặc định của mảng
                    # tải hình ảnh bao gồm 3 kênh RGB/CMYK/Alpha
                    img = cv2.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                    #tái định hình kích thước của ảnh
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    #lưu lại ảnh
                    cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                    Image1 = Image.open(f".\\2.png") 
                      
                    # sao chép hình ảnh để hình ảnh gốc không bị ảnh hưởng 
                    Image1copy = Image1.copy() 
                    Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg") 
                    Image2copy = Image2.copy() 
                      
                    # dán hình ảnh đưa ra kích thước
                    Image1copy.paste(Image2copy, (195, 114)) 
                      
                    # lưu hình ảnh  
                    Image1copy.save("{name}.png") 
                    frame = cv2.imread("{name}.png", 1)

                    cv2.imshow("Result",frame)
                    cv2.waitKey(5000)
                break

        # giải phóng nguồn cấp dữ liệu video webcam đã được tải vào bộ nhớ.
        cap.release()
        # Đóng tất cả các cửa sổ (Nếu có cửa sổ đang mở và đang chạy).
        cv2.destroyAllWindows()
        
