[index,det.box_2d, det.detected_class,proj_matrix,dim,alpha,theta_ray]







## 环境

用anacoda造的环境，cuda11，然后装cudnn就行。总共两个环境，一个yolo3d（python3.8）,一个image-g-tran（python3.9）

### yolo3d

certifi             2024.8.30
charset-normalizer  3.4.0
colorama            0.4.6
contourpy           1.1.1
cycler              0.12.1
fonttools           4.54.1
idna                3.10
importlib_resources 6.4.5
kiwisolver          1.4.7
matplotlib          3.2.2
mkl-fft             1.3.8
mkl-random          1.2.4
mkl-service         2.4.0
numpy               1.24.4
opencv-python       4.10.0.84
packaging           24.1
pandas              1.1.4
Pillow              7.1.2
pip                 24.2
pyparsing           3.1.4
python-dateutil     2.9.0.post0
pytz                2024.2
PyYAML              6.0.2
requests            2.32.3
scipy               1.10.1
seaborn             0.11.0
setuptools          75.1.0
six                 1.16.0
torch               1.8.1+cu111
torchvision         0.9.1+cu111
tqdm                4.66.6
typing_extensions   4.12.2
tzdata              2024.2
urllib3             2.2.3
wheel               0.44.0
zipp                3.20.2

### image-g-tran

attrs                     24.2.0
blinker                   1.9.0
cachetools                5.5.0
certifi                   2024.8.30
charset-normalizer        3.4.0
click                     8.1.7
colorama                  0.4.6
filelock                  3.16.1
fsspec                    2024.10.0
gitdb                     4.0.11
GitPython                 3.1.43
huggingface-hub           0.26.2
idna                      3.10
Jinja2                    3.1.4
jsonschema                4.23.0
jsonschema-specifications 2024.10.1
markdown-it-py            3.0.0
MarkupSafe                3.0.2
mdurl                     0.1.2
mpmath                    1.3.0
narwhals                  1.13.5
networkx                  3.2.1
numpy                     2.0.2
opencv-python             4.10.0.84
packaging                 24.2
pandas                    2.2.3
pillow                    11.0.0
pip                       24.2
protobuf                  5.28.3
pyarrow                   18.0.0
pydeck                    0.9.1
Pygments                  2.18.0
python-dateutil           2.9.0.post0
pytz                      2024.2
PyYAML                    6.0.2
referencing               0.35.1
regex                     2024.11.6
requests                  2.32.3
rich                      13.9.4
rpds-py                   0.21.0
safetensors               0.4.5
setuptools                75.1.0
six                       1.16.0
smmap                     5.0.1
streamlit                 1.40.1
sympy                     1.13.1
tenacity                  9.0.0
tokenizers                0.20.3
toml                      0.10.2
torch                     2.5.1
tornado                   6.4.1
tqdm                      4.67.0
transformers              4.46.2
typing_extensions         4.12.2
tzdata                    2024.2
urllib3                   2.2.3
watchdog                  6.0.0
wheel                     0.44.0

## how to run

cd 到 whole_project 

地下这些都是默认的参数文件夹，你要是改，可以从cmd，或者在代码里改，路径是 whole_project\YOLO3Dmain\inference.py,下面的这个函数

```python
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
```

在{whole_project\Image_Caption_Generator_With_Transformers-main\inference.py}中修改input和output代码

```python
input_path = "D:\\study\\whole_project\\output"
final_path = "D:\\study\\whole_project\\final"
```

然后cmd


```bash
conda activate yolo3d & python {whole_project\YOLO3Dmain\inference.py的绝对路径} & conda deactivate & conda  activate image-g-tran & python {whole_project\Image_Caption_Generator_With_Transformers-main\inference.py的绝对路径}
```

## 结果

最终结果的pkl是这样：

```json
{图片名字：[[index,   det.box_2d,    det.detected_class,   proj_matrix,  dim,alpha,theta_ray,description,score],…………]}
```



## 流程

1、用yolo3d，根据calib参数，得到3dbound的数值，同时切割成2d框用于接下来的caption，切割的小图片放在output文件夹内，output文件夹内的每个文件夹以图片名字命名(.png 替换成了-png)，内部存放分割的图片切片。

![image-20250318151524192](assets\image-20250318151524192.png)

![image-20250318151726283](assets\image-20250318151726283.png)

2、用openai的CLip模型，产生图片的描述。

3、将caption放到3dbound的框内。

![image-20250318152150210](assets\image-20250318152150210.png)

![66d83577c0d5e0bf4f512cc76bdf04e](assets\66d83577c0d5e0bf4f512cc76bdf04e.jpg)



![image-20250318152217085](assets\image-20250318152217085.png)

