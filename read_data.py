import os
from Config import config
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

con = config()


def batch_read_data(train_image, train_label):
    train_image = np.array(train_image)
    train_label = np.array(train_label)
    print("train_image.shape:{}".format(train_image.shape))
    print("train_label.shape:{}".format(train_label.shape))
    data_len = np.array(train_image).shape[0]
    num_batch = int((data_len - 1) / con.batch_size)
    print("num_batch:{}".format(num_batch))
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = train_image[indices]
    y_shuffle = train_label[indices]
    for i in range(num_batch):
        start_id = i * con.batch_size
        end_id = min((i + 1) * con.batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
def read_data():
    path = con.path
    if os.path.exists(path):
        file_name = os.listdir(path)
        print(file_name)
        Image = []
        labels = []
        for fn in file_name:
            print("fn:{}".format(fn))
            fn_path = os.path.join(path, fn)
            image = cv2.imread(fn_path)
            #cv2.imshow("image", image)
            blur_image = cv2.GaussianBlur(image, (5, 5), 0)
            hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

            # 蓝色区域的HSV值范围
            hsv_low = [105, 80, 80]
            hsv_high = [125, 180, 140]
            kernel_size = (20, 20)
            x, y, w, h = crop_image(hsv_image, hsv_low, hsv_high, kernel_size)
            image_crop = image[y:y+h, x:x+w]
            #cv2.imshow("image_crop", image_crop)
            #print("image_crop.shape:{}".format(image_crop.shape))
            image_resize = cv2.resize(image_crop, (256, 256))
            #cv2.imshow("image_resize", image_resize)

            brighter_image = brighter(image_resize, percetage=1.5)
            #cv2.imshow("brighter_image", brighter_image)

            Image.append(brighter_image)
            fn = fn.strip()
            fn_name = fn.split('-')
            #print(fn_name)
            lb = [float(fn_name[1]), float(fn_name[3]),float(fn_name[5]),float(fn_name[7])]
            labels.append(lb)
            #print(lb)
        train_image, test_image, train_label, test_label = train_test_split(Image, labels, train_size=0.9, random_state=0)
        return train_image, test_image, train_label, test_label

def crop_image(image, lower_color, upper_color, kernel_size):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
    mask = cv2.inRange(image, lower, upper)
    # 进行腐蚀和膨胀
    if kernel_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(dilated, kernel)
        #cv2.imshow("erode", eroded)

        im, contours, hierarchy= cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.imshow("im", im)
        for i in range(0, 1):
            x, y, w, h = cv2.boundingRect(contours[i])
            image = image[y:y+h, x: x+w]
            #cv2.imshow("image_crop", image)
            return x, y, w, h

def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]

    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy



if __name__ == "__main__":

    train_image, test_image, train_label, test_label = read_data()
    train_image = np.array(train_image)
    test_image = np.array(test_image)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    np.save('train_image.npy', train_image)
    np.save('test_image.npy', test_image)
    np.save('train_label.npy', train_label)
    np.save('test_label.npy', test_label)
    print("train_image.shape:{}".format(np.array(train_image).shape))
    print("train_label.shape:{}".format(np.array(train_label).shape))
