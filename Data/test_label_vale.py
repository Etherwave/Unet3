import cv2
import numpy as np

def get_special_label_image(image, id=1):
    def f(x):
        if x==id:
            return x
        return 0
    shape = image.shape
    image = list(np.array(image).flatten())
    image = list(map(f, image))
    image = np.array(image).reshape(shape)
    image = np.uint8(image)
    return image

if __name__ == '__main__':
    label_path = "../image/labels/0001TP_006690_P.png"
    label_image = cv2.imread(label_path)
    image = get_special_label_image(label_image, 17)
    print(image.shape)
    cv2.imshow("1", image)
    cv2.waitKey(1)
    rows = 720
    cols = 960
    c = 3
    flag = 0
    for i in range(rows):
        for j in range(cols):
            if image[i][j][2]>0:
                print(image[i][j])
                flag = 1
                break
        if flag>0:
            break