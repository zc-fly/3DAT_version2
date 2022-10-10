import os
import sys
import time
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def detect_predict(image):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=21)

    # load train weights
    weights_path = "fastercnn/save_weights/fasterrcnn_voc2012.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    print(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)

    # read class_indict
    label_json_path = 'fastercnn/pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    original_img = image

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        
        print(predictions)
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        # 过滤掉低概率的目标
        idxs = np.greater(predict_scores, 0.1)
        boxes = predict_boxes[idxs]
        classes = predict_classes[idxs]
        scores = predict_scores[idxs]

        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        for box, cls, score in zip(boxes, classes, scores):
            if cls == 15:
                return box



if __name__ == '__main__':
    # load image
    original_img = Image.open("test/004545.jpg").convert('RGB')
    box = detect_predict(original_img)
