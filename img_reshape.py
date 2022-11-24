
from PIL import Image
import os
from torchvision import transforms

def image_resize(image_path, new_path):           # 统一图片尺寸
    print('============>>修改图片尺寸')
    transform_list = [transforms.Resize((1024,1024)),]
                      #transforms.ToPILImage()]
    transform_img = transforms.Compose(transform_list)
    for img_name in os.listdir(image_path):
        print(img_name)
        img_path = image_path + "/" + img_name    # 获取该图片全称
        image = Image.open(img_path)# 打开特定一张图片
        if(len(image.split())==3):
            print(len(image.split()))
            print(image.size)
            image = transform_img(image)
        else:
            print(len(image.split()))
            image = image.convert("RGB")
            image = transform_img(image)

       # image = image.resize((512, 512))          # 设置需要转换的图片大小
        # process the 1 channel image
        print("----------")
        print(len(image.split()))
        print(image.size)
        # print(len(image.split()))
        image.save(new_path + '/'+ img_name)
    print("end the processing!")

   #这里ndarray_image为原来的numpy数组类型的输入




if __name__ == '__main__':
    print("ready for ::::::::  ")
    ori_path = "./BreCaHAD/images/PNG"
    new_path = "./BreCaHAD/images_reshape"                   # resize之后的文件夹路径
    image_resize(ori_path, new_path)

