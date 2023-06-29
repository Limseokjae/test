import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import predict

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.ops import box_convert
import supervision as sv
from functools import partial
import shutil
# classifier
import yaml
import timm
import pickle
from timm.models.helpers import load_checkpoint
from timm.data import create_dataset, create_loader, resolve_data_config

def load_image(image_path):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

class ARGS(object):
    def __init__(self, args):          
        for key in args:
            setattr(self, key, args[key])

def load_model(args, model_checkpoint_path, device):
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, img_name):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{img_name}_mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, f'{img_name}mask.json'), 'w') as f:
        json.dump(json_data, f)

class SAMVIT:
    def __init__(self):
        weights_path = "/home/work/Vision1/Butteryam/Weights"
        self.config = "./model/config.py"
        self.sam_args = SLConfig.fromfile(self.config)
        self.grounded_checkpoint = self.sam_args.grounded_checkpoint
        self.sam_checkpoint = self.sam_args.sam_checkpoint
        self.text_prompt = self.sam_args.text_prompt
        self.output_dir = self.sam_args.output_dir
        vit_args_path = self.sam_args.vit_args_path
        with open(vit_args_path, 'r') as f:
            vit_args = yaml.safe_load(f)

         
        self.vit_args = ARGS(vit_args)

        class_map_path = os.path.join(weights_path, "label.pkl")
        if not os.path.exists(class_map_path):
            shutil.copy(self.vit_args.class_map, class_map_path)
        self.vit_args.class_map = class_map_path

        vit_weight_path = os.path.join(weights_path, "model_best.pth.tar")
        if not os.path.exists(vit_weight_path):
            src_path = str(Path(vit_args_path).parent / "model_best.pth.tar")
            shutil.copy(src_path, vit_weight_path)
        self.vit_args.checkpoint = vit_weight_path
                
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.device = torch.device("cuda")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.vit_args.class_map, 'rb') as f:
            labels = pickle.load(f)

        self.labels_dic = {v:k for k, v, in labels.items()}

    def load_model(self):
        self.model = load_model(
            self.sam_args,
            self.grounded_checkpoint,
            self.device
        )
        # self.predictor = SamPredictor(build_sam(
        #     checkpoint=self.sam_checkpoint
        # ).to(self.device))

        self.classifier = timm.models.create_model(
            self.vit_args.model,
            pretrained=False,
            num_classes = self.vit_args.num_classes
            )
        
        self.classifier = self.classifier.to(self.device)

        load_checkpoint(self.classifier, self.vit_args.checkpoint)

        self.transform = timm.data.create_transform(
            **timm.data.resolve_data_config(self.classifier.pretrained_cfg)
        )

    
    def run_model(self, img_path):
        img_dir = Path(img_path).parent
        img_name = Path(img_path).stem

        image_source, image = load_image(img_path)

        img = image_source.copy()

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        detected_img_dir = os.path.join(img_dir, Path(img_path).stem)
        Path(detected_img_dir).mkdir(parents=True, exist_ok=True)

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        #detections = sv.Detections(xyxy=xyxy)


        cropped_imgs = []
        for img_idx, (x1, y1, x2, y2) in enumerate(xyxy):
            cropped_img = image_source[int(y1):int(y2), int(x1):int(x2)]
            cropped_imgs.append(self.transform(Image.fromarray(cropped_img)))

        input = torch.stack(cropped_imgs, dim=0)

        amp_dtype = torch.bfloat16 if self.vit_args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=self.device.type, dtype=amp_dtype)

        self.classifier.eval()
        with torch.no_grad():

            with amp_autocast():
                input = input.to(self.device)            
                output = self.classifier(input)

            # top5
            cls_logits, cls_idxs = torch.nn.functional.softmax(output).topk(5, 1)


        cls_logits = cls_logits.cpu().numpy()
        cls_idxs = cls_idxs.cpu().numpy()

        for xyxy_, logits, box_, cls_idx in zip(xyxy, cls_logits, boxes, cls_idxs):  
            cls_phrase = self.labels_dic[int(cls_idx[0])]
            print(img_path, xyxy_, cls_phrase)
