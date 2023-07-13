import os
import glob
import gradio as gr
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw
from app.utils.tools import box_prompt, format_results, point_prompt
from app.utils.tools_gradio import fast_process
from tqdm import tqdm
import cv2
from PIL import Image

cur_dir = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("seg working on the : {}".format(device))
# Load the pre-trained model
sam_checkpoint = "D:\\workplace\\MobileSAM-master\\weights\\mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)

# Description
title = "<center><strong><font size='8'>Faster Segment Anything(MobileSAM)<font></strong></center>"

description_e = """This is a demo of [Faster Segment Anything(MobileSAM) Model](https://github.com/ChaoningZhang/MobileSAM).

                   We will provide box mode soon. 

                   Enjoy!
                
              """

description_p = """ # Instructions for point mode

                0. Restart by click the Restart button
                1. Select a point with Add Mask for the foreground (Must)
                2. Select a point with Remove Area for the background (Optional)
                3. Click the Start Segmenting.

              """


#file Folder
file_pth = "D:\\workplace\\dataset\\20230423"
label_files = glob.glob(os.path.join(file_pth,"*\*.txt"),recursive=True)
img_files = glob.glob(os.path.join(file_pth,"*\*.jpg"),recursive=True)

promt_info = []
for label_file in label_files:
    with open(label_file,'r') as fr:
        annos = [eval(x.strip()) for x in fr.readlines()]
        promt_info.append(annos)


examples = []
for img in img_files:
    examples.append([img])

default_example = examples[0]


@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    return fig


def segment_with_points(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image

    # print(scaled_points, scaled_points is not None)
    # print(scaled_point_label, scaled_point_label is not None)

    nd_image = np.array(image)
    predictor.set_image(nd_image,image_format="RGB")
    masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=True,
    )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations])

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig, image


def segment_with_box(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
    box = None
):
    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    if box == None or len(box)==0:
        return image,image
    else:
        scaled_box = np.array(
            [[int(x * scale) for x in bb] for bb in box]
        )
    nd_image = np.array(image)
    box = np.array(box)
    predictor.set_image(nd_image,image_format="BGR")
    # draw_bbox(nd_image,box[0])
    # draw_bbox(nd_image,scaled_box[0])
    #only one batch predict
    all_annotations = []
    for scaled_b in scaled_box:
        masks, scores, logits = predictor.predict(box=scaled_b,multimask_output=True)
        # results = format_results(masks, scores, logits, 0)
        annotations, _ = box_prompt(masks=masks,bbox=scaled_b,target_height=new_h,target_width=new_w)
        all_annotations.append(np.array([annotations]))
    all_annotations = np.array(all_annotations)[:,0]
    fig = fast_process(
        annotations=all_annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    # return fig, None
    return fig, image


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image


def draw_bbox(image,bbox=None,point=None):
    """
    image : np
    bbox : xyxy
    """
    cv2.rectangle(image,[bbox[0],bbox[1]],[bbox[2],bbox[3]],color=(255,0,0),thickness=1)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##seg from label##
add_mask = True
promt_type = 'box' #[point or box]
max_num = 100
global_points = []
global_point_label = []
os.makedirs(os.path.join(cur_dir,"orign_images"),exist_ok=True)
os.makedirs(os.path.join(cur_dir,f"{promt_type}_images"),exist_ok=True)

for i,(img,promt) in tqdm(enumerate(zip(img_files[:max_num],promt_info[:max_num]))):
    img_name = img.split("\\")[-1].split(".")[0]
    img = Image.open(img)
    boxes = []
    for p in promt:
        xyxy = p['rect'].split(',')
        xyxy = [int(float(coord)) for coord in xyxy]
        box_center = [(xyxy[0]+xyxy[2])//2,(xyxy[1]+xyxy[3])//2]
        global_points.append(box_center)
        global_point_label.append(add_mask)
        boxes.append(xyxy)
    if promt_type=='point':
        fig,re_img = segment_with_points(img)
    elif promt_type=='box':
        fig,re_img = segment_with_box(img,box=boxes)
    cv2.imwrite(os.path.join(os.path.join(cur_dir,"orign_images",img_name+'.jpg')),np.array(re_img)[...,::-1])
    cv2.imwrite(os.path.join(os.path.join(cur_dir,f"{promt_type}_images",img_name+'.jpg')),np.array(fig)[...,::-1])