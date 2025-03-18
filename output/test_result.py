import pickle
with open('D:\\study\whole_project\\output\\result_3d_dic.pkl', 'rb') as f:  # 'rb' 表示以二进制读取模式打开文件
    loaded_data = pickle.load(f)


for img_name,info in loaded_data.items():
    print(img_name,loaded_data[img_name][1][1])
