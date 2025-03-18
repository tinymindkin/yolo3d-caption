
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from pathlib import Path
import pickle
import os
import cv2
from torch.nn import functional as F
input_path = "D:\\study\\whole_project\\output"
final_path = "D:\\study\\whole_project\\final"


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output = model.generate(pixel_values,output_scores=True,return_dict_in_generate=True, **gen_kwargs) ## get lots dic
  output_ids = output['sequences']
  description_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  description_preds = [pred.strip() for pred in description_preds]


  last_step_logits = output['scores'][-1]  # last logit
  probs = F.softmax(last_step_logits, dim=-1)  # 概率分布
  max_prob, _ = probs.max(dim=-1)  # 最大概率
  values_list = max_prob.tolist()
  final_dic = {"prediction":description_preds,"scores":values_list}
  print(">>>>>>>>>>>>>>>描述",f"description:{description_preds[0]}")
  print(">>>>>>>>>>>>>>>评分",f"score:{values_list[0]}")
  return final_dic



def draw_text_in_rectangle(img, top_left, bottom_right, text):
    # width and heitght
    rect_width = bottom_right[0] - top_left[0]
    rect_height = bottom_right[1] - top_left[1]
    
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # text size 
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # # 
    # while text_size[0] > rect_width or text_size[1] > rect_height:
    #     font_scale -= 0.1
    #     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # center 
    text_x = top_left[0] + (rect_width - text_size[0]) // 2
    text_y = top_left[1] + (rect_height + text_size[1]) // 2

    # draw 
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)





with open(input_path + '\\result_3d_dic.pkl',mode="rb") as f:
  dic_data = pickle.load(f)
for img_name,info_3d in dic_data.items():
    img_dir = os.path.join(input_path , img_name.replace(".","-"))
    final_img_dir = os.path.join(final_path , img_name.split(".")[0] + ".jpg")
    img = cv2.imread(final_img_dir)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>",final_img_dir)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>",img_dir)
    for index,bound_info in enumerate(info_3d):
      segment_path = os.path.join(img_dir,f"{bound_info[0]}.jpg")  
      dic_data[img_name][index].append(predict_step([segment_path])["prediction"][0]) ## ["prediction"]
      dic_data[img_name][index].append(predict_step([segment_path])["scores"][0]) ## ["prediction"]
      # draw description
      draw_text_in_rectangle(img, top_left=bound_info[1][0], bottom_right=bound_info[1][1], text = predict_step([segment_path])["prediction"][0]) ##["prediction"]
    cv2.imwrite(final_img_dir, img)
        


with open( final_path + "\\final_dic.pkl",mode="wb") as f:
  pickle.dump(dic_data, f)