import cv2
import numpy as np
import os

def get_special_label_image(image, id=1):
    image = image[:, :, 0]
    def f(x):
        if x==id:
            return 1
        return 0
    shape = image.shape
    image = list(np.array(image).flatten())
    image = list(map(f, image))
    image = np.array(image).reshape(shape)
    image = np.uint8(image)
    return image

def build_folder(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

def build_new_label_only_road():
    old_label_path = "../image/labels"
    new_label_path = "../image/new_labels"

    build_folder(new_label_path)

    labels = [f for f in os.listdir(old_label_path) if f.endswith("png")]

    class_no = 17

    for i in range(len(labels)):
        old_label_image_path = old_label_path+"/"+labels[i]
        old_image = cv2.imread(old_label_image_path)
        new_image = get_special_label_image(old_image, class_no)
        # 虽然我们已经处理成了一维图片，但写成jpg的时候又会搞成3通道
        #print(new_image.shape)
        new_label_image_path = new_label_path+"/"+labels[i]
        cv2.imwrite(new_label_image_path, new_image)
        if i%10==0:
            print("已完成{}/{}".format(i, len(labels)))

def test_new_label_image():
    new_label_image_folder = "../image/new_labels"
    image_path = new_label_image_folder+"/"+"0001TP_006690.png"
    image = cv2.imread(image_path)
    print(image.shape)
    cv2.imshow("1", image)
    cv2.waitKey()

def rename_new_label_image():
    '''
    之前没发现image的名字没有_p，但是label名字多了个_p写到dataloader才发现，去吧new_label重命名一下，去掉_p
    :return:
    '''
    new_label_path = "../image/new_labels"
    labels = os.listdir(new_label_path)
    print(len(labels))
    for i in range(len(labels)):
        src = new_label_path+"/"+labels[i]
        dst = new_label_path+"/"+labels[i][:-6]+".png"
        os.rename(src, dst)
        if i%10==0:
            print("已完成{}/{}".format(i, len(labels)))


if __name__ == '__main__':
    # build_new_label_only_road()
    test_new_label_image()
    # rename_new_label_image()

