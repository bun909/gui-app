import cv2
import os

def start_capture(name):
        # setup nơi lưu trữ dữ liệu
        path = "./data/" + name
        # khởi tạo số lượng hình ảnh ban đầu = 0
        num_of_images = 0
        # Tải tệp xếp tầng để phát hiện khuôn mặt
        detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
        #khởi tạo thư mục lưu trữ
        try:
            os.makedirs(path)
        except:
            print('Directory Already Created')
        # Khởi tạo nguồn cấp dữ liệu webcam
        vid = cv2.VideoCapture(0)
        # khởi tạo một vòng lặp while chạy vô thời hạn hoặc cho đến khi tất cả các khung được lặp qua. 
        # Mã kèm theo bên trong được thực thi cho mỗi khung hình đến từ webcam. 
        # Vòng lặp cũng có thể bị phá vỡ nếu người dùng nhấn phím ‘q’
        while True:
            # tìm nạp các khung hình liên tiếp từ nguồn cấp dữ liệu webcam
            ret, img = vid.read()
            new_img = None
            # chuyển đổi ảnh sang thang độ xám
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Tìm kiếm các khuôn mặt trong hình ảnh bằng cách sử dụng tệp xếp tầng đã tải
            face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            # Nếu các khuôn mặt được tìm thấy, nó sẽ trả về vị trí 
            # của các khuôn mặt được phát hiện dưới dạng Rect(x, y, w, h)
            for x, y, w, h in face:
                # Vẽ một hình chữ nhật với đường viền màu đen có độ dày 2 px
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                # ghi nội dung "Nhan Dien Khuon Mat" (màu xanh dương) lên đầu khung chữ nhật vừa vẽ
                cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                # ghi nội dung "images captured" (màu xanh dương) lên đầu khung chữ nhật của hình chụp + thứ tự tăng dần
                cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                #trả về khuôn mặt mới đã cắt từ hình ảnh gốc
                new_img = img[y:y+h, x:x+w]
            cv2.imshow("FaceDetection", img)
            
            key = cv2.waitKey(1) & 0xFF
            #tạo ra ảnh theo thứ tự tên tăng dần
            try :
                cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
                num_of_images += 1
            except :

                pass
            # đợi phím 'q' được nhấn hoặc nhấn phím esc hoặc đợi số ảnh chụp lớn hơn 310 ảnh
            if key == ord("q") or key == 27 or num_of_images > 310:
                break
        cv2.destroyAllWindows()
        return num_of_images

