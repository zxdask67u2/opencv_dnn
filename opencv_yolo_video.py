import numpy as np
import cv2 as cv
import time
import threading


weightsPath = "./yolov3.weights"  # 权重文件
configPath =  "./yolov3.cfg"# 配置文件
labelsPath =  "./coco.names" # label名称
imgPath =  '86.jpg' # 测试图像
CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.4  # 非最大值抑制阈值


with open(labelsPath, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')

## 加载下载的文件
net = cv.dnn.readNetFromDarknet(configPath, weightsPath)



class camera():#摄像头读取
    def __init__(self):
        self.frame=[]
    def read_video(self,path):
        vdo = cv.VideoCapture(path)
        # vdo=cv2.VideoCapture(0)
        while 1:
            ret, frame = vdo.read()
            # print('111')
            if ret:
                # print('read frame and update')
                # self.frame=frame
                self.frame = cv.resize(frame, (640, 480))
            else:
                print('丢失相机')
                vdo = cv.VideoCapture(path)


Camera1=camera()
def c1():
    global Camera1
    Camera1.read_video('rtsp://admin:epc2019.@192.168.0.64')  #启动循环
    # Camera1.read_video(0)

def main1():
    global Camera1

    # Start training
    while 1:
        if Camera1.frame==[]:
            continue
        t1=time.time()

        #-----------------------------------------------
        img =Camera1.frame.copy()
        blobImg = cv.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True,
                                       False)  ## net需要的输入是blob格式的，用blobFromImage这个函数来转格式
        net.setInput(blobImg)  ## 调用setInput函数将图片送入输入层

        # 获取网络输出层信息（所有输出层的名字），设定并前向传播
        outInfo = net.getUnconnectedOutLayersNames()  ## 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
        layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
        # print(layerOutputs)

        (H, W) = img.shape[:2]

        # 过滤layerOutputs
        # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
        # 过滤后的结果放入：
        boxes = []  # 所有边界框（各层结果放一起）
        confidences = []  # 所有置信度
        classIDs = []  # 所有分类ID

        # # 1）过滤掉置信度低的框框
        for out in layerOutputs:  # 各个输出层
            for detection in out:  # 各个框框
                # 拿到置信度
                scores = detection[5:]  # 各个类别的置信度
                classID = np.argmax(scores)  # 最高置信度的id即为分类id
                confidence = scores[classID]  # 拿到置信度

                # 根据置信度筛查
                if confidence > CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
        idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # boxes中，保留的box的索引index存入idxs

        with open(labelsPath, 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')

        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                                   dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
        if len(idxs) > 0:
            for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
                if classIDs[i]==0:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
                    text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                    cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color,
                               2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px
        cv.imshow('目标检测结果', img)

        #-----------------------------------------------


        print(1/(time.time()-t1))
        if cv.waitKey(1)==27:
            break

if __name__ == '__main__':
    t1=threading.Thread(target=c1)
    t2=threading.Thread(target=main1)
    t1.start()
    t2.start()