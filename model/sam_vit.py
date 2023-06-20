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

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.ops import box_convert
import supervision as sv
from functools import partial

# classifier
import yaml
import timm
import pickle
from timm.models.helpers import load_checkpoint
from timm.data import create_dataset, create_loader, resolve_data_config

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

class ARGS(object):
    def __init__(self, args):          
        for key in args:
            setattr(self, key, args[key])

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
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
        self.config = "./model/config.py"
        vit_args_path = "./model/vit_args.yaml"
        self.grounded_checkpoint = "./model/groundingdino_swint_ogc.pth"
        self.sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
        self.text_prompt = 'foods, tools, cutlery, plate, bowl, frypan, pot, cooking pot'
        self.output_dir = './output'
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.device = torch.device("cuda")
        
        with open(vit_args_path, 'r') as f:
            vit_args = yaml.safe_load(f)

        self.vit_args = ARGS(vit_args)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.vit_args.class_map, 'rb') as f:
            labels = pickle.load(f)

        self.labels_dic = {v:k for k, v, in labels.items()}

    def load_model(self):
        self.model = load_model(
            self.config,
            self.grounded_checkpoint,
            self.device
        )
        # self.predictor = SamPredictor(build_sam(
        #     checkpoint=self.sam_checkpoint
        # ).to(self.device))

        self.classifier = timm.models.create_model(
            self.vit_args.model,
            pretrained=False,
            num_classes = 56
            )
        
        self.classifier = self.classifier.to(device)
        load_checkpoint(self.classifier, self.vit_args.checkpoint)
    
    def run_model(self, img_path):
        img_dir = Path(img_path).parent
        img_name = Path(img_path).stem
        image_pil, image = load_image(img_path)
        boxes_filt, pred_phrases = get_grounding_output(
            self.model,
            image,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold,
            self.device
        )

        detected_img_dir = os.path.join(img_dir, Path(img_path).stem)
        Path(detected_img_dir).mkdir(parents=True, exist_ok=True)

        h, w, _ = image_pil.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)

        for img_idx, (x1, y1, x2, y2) in enumerate(xyxy):
            cv2.imwrite(os.path.join(detected_img_dir, f"{img_idx:05d}.png"), image_pil[int(y1):int(y2), int(x1):int(x2), ::-1])


        dataset = create_dataset(
            root=detected_img_dir,
            name='',
            search_split=False,
        )
        data_config = resolve_data_config(
            vars(self.vit_args),
            model=self.classifier,
            use_test_size=True,
            verbose=True,
        )

        loader = create_loader(
            dataset,
            input_size=data_config['input_size'],
            batch_size=10,
            use_prefetcher=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=self.vit_args.workers,
            crop_pct=data_config['crop_pct'],
            crop_mode=data_config['crop_mode'],
            pin_memory=self.vit_args.pin_mem,
            device=self.device,
            tf_preprocessing=False,
        )


        amp_dtype = torch.bfloat16 if self.vit_args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=self.qdevice.type, dtype=amp_dtype)

        self.classifier.eval()
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                # compute output
                with amp_autocast():
                    input = input.to(self.device)            
                    output = self.classifier(input)

                _, pred_batch = output.topk(1, 1, True, True)
                
                for idx in pred_batch.cpu().numpy():
                    print(self.labels_dic[int(idx)])

        # image = cv2.imread(img_path)
        # cv2.imwrite(os.path.join(self.output_dir, Path(img_path).name), image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # self.predictor.set_image(image)

        # size = image_pil.size
        # H, W = size[1], size[0]
        # for i in range(boxes_filt.size(0)):
        #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        #     boxes_filt[i][2:] += boxes_filt[i][:2]

        # boxes_filt = boxes_filt.cpu()
        # transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        # masks, _, _ = self.predictor.predict_torch(
        #     point_coords = None,
        #     point_labels = None,
        #     boxes = transformed_boxes.to(self.device),
        #     multimask_output = False,
        # )

        # # draw output image
        # plt.figure(figsize=(10, 10))
        # #plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)

        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(self.output_dir, f"{img_name}_grounded_sam_output.jpg"), 
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )

        # save_mask_data(self.output_dir, masks, boxes_filt, pred_phrases, img_name)
