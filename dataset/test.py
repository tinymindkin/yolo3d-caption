import numpy as np 
import cv2

content = np.load("D:\\study\\whole_project\\dataset\\nuScenes-panoptic-v1.0-all\\panoptic\\v1.0-mini\\0ab9ec2730894df2b48df70d0d2e84a9_panoptic.npz")
print(type(content))
print(content)
for key,value in content.items():
    print()
img = content["data"].reshape(224, 224)

img = np.clip(img, 0, 255)



# image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # 使用 OpenCV 显示图像
# cv2.imshow("OpenCV Image", image_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

