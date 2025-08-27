import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

MODEL_WEIGHTS = "/mnt/d/Summer_training/Week_3/resnet_24000_city_foggy_2/model_final.pth"  # 模型
CONFIG_FILE = "/mnt/d/Summer_training/Week_3/configs/foggy_resnet50.yaml"         # config
IMAGE_DIR = os.path.join("/mnt/d/Summer_training/Week_3/foggy_cityscape/foggy_cityscape/VOC2007/JPEGImages")          # 圖片資料夾
TEST_LIST = os.path.join("/mnt/d/Summer_training/Week_3/foggy_cityscape/foggy_cityscape/VOC2007/ImageSets/Main/test.txt")  # 測試集 txt
OUTPUT_DIR = "./foggy1_results_2"  # 結果

os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(CONFIG_FILE)
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)


metadata = MetadataCatalog.get("cityscapes_val")

with open(TEST_LIST, "r") as f:
    test_files = [line.strip() for line in f.readlines()]

print(f"共讀取 {len(test_files)} 張測試圖片。")

num = 1
for img_id in test_files:
    img_path = os.path.join(IMAGE_DIR, img_id + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMAGE_DIR, img_id + ".png")
        if not os.path.exists(img_path):
            print(f"找不到圖片: {img_id}")
            continue

    img = cv2.imread(img_path)
    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    save_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
    print(num)
    num += 1

