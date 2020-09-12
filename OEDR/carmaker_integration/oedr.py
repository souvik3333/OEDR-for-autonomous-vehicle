import socket
from PIL import Image
from scipy.misc import imread
import PIL
import os
import cv2
import argparse
import time
from sys import platform
from skimage import color
from skimage import io
from torchvision import transforms
import pandas as pd

from models import *
from utils.datasets import *
from utils.utils import *
from utils.curves import Curves
import trafficLight
import trafficSignModel
# """
# connecting to ipg movie server
client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
ip=socket.gethostbyname("127.0.0.1")
port=2210
address=(ip,port)
client.connect(address)
# """


# initialize the yolov3 model
# """
weight_file_path = "weights/last_40000.pt"
cfg_file_path = "cfg/yolov3-spp.cfg"
nms_thres = 0.5
conf_thres = 0.3
class_data = "data/rovit.data"
img_size = (320, 192)  if ONNX_EXPORT else 416 # (320, 192) or (416, 256) or (608, 352) for (height, width)
# Initialize
device = torch_utils.select_device()
torch.backends.cudnn.benchmark = False  # set False to speed up variable image size inference
# Initialize model
model = Darknet(cfg_file_path, img_size)

# Load weights
model.load_state_dict(torch.load(weight_file_path, map_location=device)['model'])
# _ = load_darknet_weights(model, weight_file_path)

# Fuse Conv2d + BatchNorm2d layers
# model.fuse()

# Eval mode
model.to(device).eval()

# Export mode
if ONNX_EXPORT:
    img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
    torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
    exit(0)

# Get classes and colors
classes = load_classes(parse_data_cfg(class_data)['names'])

# loading traffic sign model

fTrafficSignModel = trafficSignModel.LeNet()
fTrafficSignModel.load_state_dict(torch.load("models/test.pt"))
fTrafficSignModel.eval()


def detect(img_path):
    dataloader = LoadImages(img_path, img_size=img_size, half=False)
    result={}
    for path, img, im0, vid_cap in dataloader:
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            index=0
            for *xyxy, conf, _, cls in det:
          #      if save_img or stream_img:  # Add bbox to image
                label = '%s %.2f' % (classes[int(cls)], conf)
                # plot_one_box(xyxy, im0,label=label, color=colors[int(cls)])
                print(label)
                result[index]=(label,xyxy)
                index=index+1
    return result
# """
# loading traffic sign classes
trafficSignData = pd.read_csv("data/signnames.csv")
trafficSignData = trafficSignData["SignName"]

# func to classify traffic sign
def trafficSign(ori_img):
    ori_img = Image.fromarray(color.rgb2gray(ori_img))

    t = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = torch.autograd.Variable(t(ori_img).unsqueeze(0))

    output = fTrafficSignModel(img)
    pred = output.data.max(1, keepdim=True)[1][0][0]
    return trafficSignData[pred.item()] 
# """
# lane detection pipeline

def pipeline(img):    
    left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = findLines(img)
    yRange = 719
    leftCurvature = calculateCurvature(yRange, left_fit_m) / 1000
    rightCurvature = calculateCurvature(yRange, right_fit_m) / 1000
    output = drawLine(img, left_fit, right_fit)
    cv2.putText(output, 'Left Radius:', (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    cv2.putText(output, str(leftCurvature), (220,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    cv2.putText(output, 'Right Radius:', (10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    cv2.putText(output, str(rightCurvature), (220,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    return cv2.cvtColor( output, cv2.COLOR_BGR2RGB )

# depth estimation of each object
def depth_estimation(imgL,imgR,x_left,x_right,y_top,y_bottom):
    # # Reading Left and Right images

    im_width=640
    im_height=480

    # Computing disparity
    stereo = cv2.StereoSGBM_create(numDisparities=320	, blockSize=11)
    disparity = stereo.compute(imgL,imgR)
    

    # Calulating depth map from disparity map
    nDisparity = np.array(disparity)+17
    nDisparity = 29184/(nDisparity*2)
    x_left = max(0,x_left-(x_right-x_left))
    x_right = min(640,x_right+(x_right-x_left))
    return np.amin(nDisparity[y_top:y_bottom,x_left:x_right])

# """
# reading IPG movie server name
data = client.recv(64)

# testing 
itr = 0
imageFrames = []
while True:
    # 0
    header = client.recv(64)
    headerStr=""
    try:
       headerStr=header.decode('utf-8') 
    except:
        continue
    if "921600" in headerStr and "640x480" in headerStr:
        data0 = client.recv(921600)
        headerDataList0 = header.decode('utf-8').split(' ')
        channel = headerDataList0[1]
        sizeList0 = headerDataList0[4].split('x')

        if(channel=='0'):
            fnameName0="name"+str(headerDataList0[1])+str(headerDataList0[3])+".ppm"
            imgHeader0="P6\n"+str(sizeList0[0])+" "+str(sizeList0[1])+"\n255\n"
        else:
            continue

        
    else:
        tmp = client.recv(921600-len(header.decode('utf-8')))
        continue

    # 1

    header = client.recv(64)
    headerStr=""
    try:
       headerStr=header.decode('utf-8') 
    except:
        continue
    if "921600" in headerStr and "640x480" in headerStr:
        data = client.recv(921600)
        headerDataList = header.decode('utf-8').split(' ')
        channel = headerDataList[1]
        sizeList = headerDataList[4].split('x')

        if(channel=='1'):
            fnameName="name"+str(headerDataList[1])+str(headerDataList[3])+".ppm"
            imgHeader="P6\n"+str(sizeList[0])+" "+str(sizeList[1])+"\n255\n"
        else:
            continue
    else:
        tmp = client.recv(921600-len(header.decode('utf-8')))
        continue

    with open(fnameName,"wb+") as f:
        f.write(bytes(imgHeader, 'utf-8'))
        f.write((data))

    with Image.open(fnameName) as image:
        image.save(fnameName[:-3]+"jpg")
    os.remove(fnameName)

    # depth code
    # 
    with open(fnameName0,"wb+") as f:
        f.write(bytes(imgHeader0, 'utf-8'))
        f.write((data0))
    with Image.open(fnameName0) as image:
        image.save(fnameName0[:-3]+"jpg")
    os.remove(fnameName0) 
    imgL = cv2.imread(fnameName0[:-3]+"jpg")
    imgR = cv2.imread(fnameName[:-3]+"jpg")
    # print("Depth:"+str(depth_estimation(imgL,imgR,0,640,0,480)))
    # 

    # img = cv2.imread(fnameName0[:-3]+"jpg")
    # cv2.imshow('Test image',img)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
    
    # read image in plt format
    # img = plt.imread(fnameName[:-3]+"jpg")
    img = Image.open(fnameName[:-3]+"jpg")
    img = img.resize((1280,720))
    img.save("1"+fnameName[:-3]+"jpg")
    img.close()
    os.remove(fnameName[:-3]+"jpg")
    img = plt.imread("1"+fnameName[:-3]+"jpg")
    # get result from yolo module
    img = pipeline(img)
    result=detect("1"+fnameName[:-3]+"jpg")
    # detect traffic sign, traffic ligth
    for i in result:
        classAcc= result[i][0].split(" ")

        mxyxy = (result[i][1][0].item(),result[i][1][1].item(),result[i][1][2].item(),result[i][1][3].item())
        
        depthVal = str(np.ceil(depth_estimation(imgL,imgR,int(mxyxy[0]/2),int(mxyxy[2]/2),int(mxyxy[1]*(2/3)),int(mxyxy[3]*(2/3)))))

        print(classAcc[0])

        # if(classAcc[0]=="stop"):
        #     # call traffic signal code
        #     print(trafficSign(img[int(mxyxy[1]):int(mxyxy[3]),int(mxyxy[0]):int(mxyxy[2])]))
        if(classAcc[0]=="trafficlight"):
            # call traffic light code
            trafficLightMat=(trafficLight.estimate_label(img[int(mxyxy[1]):int(mxyxy[3]),int(mxyxy[0]):int(mxyxy[2])]))
            if trafficLightMat==[1,0,0]:
                plot_one_box(mxyxy,img,label=result[i][0],additional_label_inf="RED")
            elif trafficLightMat==[0,1,0]:
                plot_one_box(mxyxy,img,label=result[i][0],additional_label_inf="Yellow")
            else:
                plot_one_box(mxyxy,img,label=result[i][0],additional_label_inf="Green")
        elif(classAcc[0]=="trafficsignal"):
            signVal = trafficSign(img[int(mxyxy[1]):int(mxyxy[3]),int(mxyxy[0]):int(mxyxy[2])])
            plot_one_box(mxyxy,img,label=result[i][0],additional_label_inf=str(signVal))
        else:
            plot_one_box(mxyxy,img,label=result[i][0])
        cv2.putText(img, 'Depth:'+depthVal, (int(mxyxy[2]),int(mxyxy[3])),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0))
    # img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # show the result
    # cv2.imshow('Test image',img)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
    # save the result
    imageFrames.append(img)
    if(itr==312):
        video = cv2.VideoWriter("result.avi", 0, 1, (1280, 720))
        for image in imageFrames:
            video.write(image)  
        cv2.destroyAllWindows()
        video.release()
        exit(0)
    
    itr=itr+1

client.close()
# """
'''
def main():
    print("Hello World!")
    img = Image.open("data/name12.171.jpg")
    img = img.resize((1280,720))
    img.save("data/name12.171.jpg")
    img = plt.imread("data/name12.171.jpg")
    img = pipeline(img)
    plt.imshow(img)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imsave("lane_final.jpg",img)
    plt.show()

if __name__ == "__main__":
    main()

'''