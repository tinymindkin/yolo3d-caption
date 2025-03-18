# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""
import pickle
import json
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
from pathlib import Path
import glob

import cv2
import torch

### 加入根目录，配置环境变量
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
ROOT = ROOT.resolve()
print(ROOT)
### add PARENT_ROOT  
PARENT_ROOT = ROOT.parents[0]  
if str(PARENT_ROOT) not in sys.path: 
	sys.path.append(str(PARENT_ROOT)) 
PARENT_ROOT = PARENT_ROOT.resolve()
print(ROOT)

## save coordination and class of img
result_3d_dic = {}

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import LOGGER, check_img_size, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device, time_sync

import torch
import torch.nn as nn
from torchvision.models import resnet18#, vgg11

import numpy as np

from script.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from script import Model, ClassAverages
from script.Model import ResNet, ResNet18, VGG11

## the blow is to check if your Gpu right
print("是否可用：", torch.cuda.is_available())        # if Gpu available
print("GPU数量：", torch.cuda.device_count())       # GPU number
print("torch方法查看CUDA版本：", torch.version.cuda)  # version of cudn by torch
print("GPU索引号：", torch.cuda.current_device())    # current_device number


# model factory to choose model
model_factory = {
    'resnet': resnet18(pretrained=True),
    'resnet18': resnet18(pretrained=True),
    # 'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

def expand_bound_with_limits(x1, y1, w, h, image_width, image_height, scale=5/4):
    # 计算中心点
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    # 计算新的宽度和高度
    w_new = scale * w
    h_new = scale * h
    
    # 计算新的左上角和右下角坐标
    x1_new = int(cx - w_new / 2)
    y1_new = int(cy - h_new / 2)
    x2_new = int(cx + w_new / 2)
    y2_new = int(cy + h_new / 2)
    
    # 确保不超过图像的边界
    # 修正左上角坐标不能小于0
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    
    # 修正右下角坐标不能超过图像宽度和高度
    x2_new = min(image_width, x2_new)
    y2_new = min(image_height, y2_new)
    
    w_new = x2_new - x1_new
    h_new = y2_new - y1_new
    return x1_new, y1_new, w_new, h_new
def detect3d(
    reg_weights,
    model_select,
    source,
    calib_dir,
    show_result,
    save_result,
    output_path,
    illusion_path
    ):

    # Directory
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    ### calib = str(calib_dir)

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # loop images
    for i, img_path in enumerate(imgs_path):
        calib_file = os.path.join(calib_dir, f'{os.path.basename(img_path).split(".")[0]}.txt')
        calib_file = str(calib_file)
        print("calib_filecalib_filecalib_file",calib_file)
        ## add img name
        result_3d_dic[os.path.basename(img_path)] = []
        # read image
        img = cv2.imread(img_path)
        pure_img = img.copy()
        ## img_height img_width
        img_height, img_width = img.shape[:2]
        ## cach_per_img
        cache_per_img = output_path / os.path.basename(img_path).replace(".","-")
        
        ## make cache_per_img 
        try:
            os.mkdir(cache_per_img)
        except Exception as e:
            print(e)   
        
        # Run detection 2d
        dets = detect2d(
            weights='D:\study\whole_project\yolov5s.pt',
            source=img_path,
            data=ROOT / 'data/coco128.yaml',
            imgsz=[640, 640],
            device=0,
            classes=[0, 2, 3, 5]
        )
        

        for index,det in enumerate(dets):
            if not averages.recognized_class(det.detected_class):
                continue
            try: 
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib_file)
            except Exception as e:
                print(">>>>>>>>>>>>>>>>>>>>>","nononononodectedobject","-----",e)
                continue
                
            
            

            theta_ray = detectedObject.theta_ray  ## 1\ theta_ray
            input_img = detectedObject.img  
            proj_matrix = detectedObject.proj_matrix  ## proj_matrix
            box_2d = det.box_2d   ## box_2d
            detected_class = det.detected_class  ## detected_class
            print("------------------",theta_ray,proj_matrix,box_2d,detected_class)

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            # predict orient, conf, and dim
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            
            ## add detected iamge infom [box_2d,detected_class]
            result_3d_dic[os.path.basename(img_path)].append([index,det.box_2d, det.detected_class,proj_matrix,dim,alpha,theta_ray])
           
            ### incise image 
            x, y, width, height = det.box_2d[0][0], det.box_2d[0][1], det.box_2d[1][0] - det.box_2d[0][0], det.box_2d[1][1] - det.box_2d[0][1]

            ### enlarge 
            
            # new_width = int(width * 1.5)
            # new_height = int(height * 1.5)
            
            # new_x = max(0, x - (new_width - width) // 2)
            # new_y = max(0, y - (new_height - height) // 2)

            new_x, new_y, new_width, new_height = expand_bound_with_limits(x, y, width, height,img_width,img_height,scale=5/4)

            
            new_x_end = new_x + new_width
            new_y_end = new_y + new_height
            

            cropped_img = pure_img[new_y:new_y_end, new_x:new_x_end]



            ## save cropped_img
            if cropped_img is not None and cropped_img.size > 0:
                cv2.imwrite(str(cache_per_img / f"{index}.jpg").replace("\\","\\\\"), cropped_img)
            else:
                print("Error: cropped_img is empty or None.")

            #plot 3d detection
            plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)  

        if show_result:
            cv2.imshow('3d detection', img)
            cv2.waitKey(0)

        ## save reult image and image info
        if save_result is not None:
            try:
                os.mkdir(os.path.join(illusion_path))
            except:
                pass
            cv2.imwrite(os.path.join(illusion_path,f'{os.path.basename(img_path).split(".")[0]}.jpg'), img)

    ### save result_3d_dic 
    if output_path is not None:
        try: 
            with open(os.path.join(output_path,'result_3d_dic.pkl'), 'wb') as f:  # 'wb' in binary mode
                pickle.dump(result_3d_dic, f)
        except Exception as e: 
            print(f"Failed to save JSON: {e}")
        


@torch.no_grad()
def detect2d(
    weights,
    source,
    data,
    imgsz,
    device,
    classes
    ):

    # array for boundingbox detection
    bbox_list = []

    # Directories
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader 
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(prediction=pred, classes=classes)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)  # integer class
                    label = names[c]

                    bbox_list.append(Bbox(bbox, label))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return bbox_list

def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    img_2d=None
    ):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default= PARENT_ROOT / 'eval/image_2' , help='file/dir/URL/glob, 0 for webcam')  ###PARENT_ROOT / 'eval/image_2' 
    parser.add_argument('--data', type=str, default=os.path.join(ROOT,'data/coco128.yaml'), help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default=ROOT / 'weights/resnet18.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet18', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_dir', type=str, default=PARENT_ROOT / 'eval/calib', help='Calibration  path')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=PARENT_ROOT / 'output', help='Save output pat')
    parser.add_argument('--illusion_path', type=str, default=PARENT_ROOT / 'final', help='illusion pat')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_dir=opt.calib_dir,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path,
        illusion_path=opt.illusion_path
    )

if __name__ == "__main__":
    print("this is root::::::::::::---)))",ROOT)
    opt = parse_opt()
    main(opt)