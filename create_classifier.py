import numpy as np
from PIL import Image
import os, cv2



# Phương pháp đào tạo bộ phân loại tùy chỉnh để nhận dạng khuôn mặt
def train_classifer(name):
# Đọc tất cả các hình ảnh trong tập dữ liệu tùy chỉnh
# getcwd () trả về thư mục làm việc hiện tại của một tiến trình
    path = os.path.join(os.getcwd()+"/data/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}


    # Lưu trữ hình ảnh ở định dạng numpy và id của người dùng trên cùng một chỉ mục trong danh sách imageNp và id
    for root,dirs,files in os.walk(path):
            pictures = files


    for pic in pictures :

            imgpath = path+pic
            # convert ảnh gốc sang ảnh mode L
            img = Image.open(imgpath).convert('L')
            #khai báo mảng và kiểu dữ liệu
            imageNp = np.array(img, 'uint8')
            #trả lại mảng các chữ số của id sau khi chia  mảng pic thành các chữ số phân tách nhau bởi mảng 0
            id = int(pic.split(name)[0])
            #imageNp phần tử muốn thêm vào vị trí cuối cùng của faces, sau khi thêm thì tổng số phần tử sẽ tăng lên một
            faces.append(imageNp)
            #id phần tử muốn thêm vào vị trí cuối cùng của ids, sau khi thêm thì tổng số phần tử sẽ tăng lên một 
            ids.append(id)

    ids = np.array(ids)

    # Đào tạo và lưu bộ phân loại
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("./data/classifiers/"+name+"_classifier.xml")

