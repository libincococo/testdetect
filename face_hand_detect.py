#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import argparse
import tensorflow as tf
import cv2
import numpy as np
import time
import sys
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# from object_detection.utils import visualization_utils as vis_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append("..")


def load_detec_graph(graph_path):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph_path, 'rb') as fid:
        graph_def.ParseFromString(fid.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


def load_detec_labe_map(labe_map_path):
    label_map = label_map_util.create_category_index_from_labelmap(labe_map_path)
    return label_map


def load_recog_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def load_recog_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def inference_for_single_image(image, graph):
    with graph.as_default():
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image
            # coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
    return tensor_dict


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    # float_caster = tf.cast(image_reader, tf.float32)
    float_caster = tf.cast(file_name, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    # sess = tf.Session()
    # result = sess.run(normalized)
    return normalized


def visualization(image, box, display_str='', thickness=4, use_normalized_coordinates=True,color='Chartreuse'):
    # Draw boxe onto image.
    # image = image_np.copy()
    ymin, xmin, ymax, xmax = box

    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))
    return image


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, thickness,
                               display_str, use_normalized_coordinates):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_height = font.getsize(display_str)[1]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height *= (1 + 2 * 0.05)

    if top > display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + display_str_height
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin

def get_roi_box(box, scale=1.1):
  top, left, bottom, right = box
  height = (bottom - top) * scale
  width = (right - left) * scale
  center = [(top + bottom) / 2.0, (left + right) / 2.0]
  new_top = center[0] - height/2.0
  if new_top < 0:
    new_top = 0
  new_left = center[1] - width/2.0
  if new_left < 0:
    new_left = 0
  new_bottom = center[0] + height/2.0
  if new_bottom > 1:
    new_bottom = 1
  new_right = center[1] + width/2.0
  if new_right >1:
    new_right = 1

  return (new_top, new_left, new_bottom, new_right)

def nothing(emp):
	pass


# playtype 0 播放摄像头， 1 播放视频文件  2 播放图片
def main(vidoefile,playtype = 0):
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"
    cap = None
    frames = 0
    frame = None
    ret = True
    waitkeytime=10

    if playtype == 0:
        isvideo = False
        isimage = False
    if playtype == 1:
        isvideo = True
        isimage = False
    if playtype >=2 :
        isvideo = False
        isimage = True

    cv2.namedWindow('gesture')
    if isimage:
        isvideo = False
        waitkeytime = 5000


    if isvideo and isimage == False:
        cap = cv2.VideoCapture(vidoefile)
        if not cap.isOpened():
            return
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        loop_flag = 0
        pos = 0
        cv2.createTrackbar('time', 'gesture', 0, frames, nothing)
        ret, frame = cap.read()

    if isvideo == False and isimage == False:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        ret, frame = cap.read()


    if isimage:
        frame = cv2.imread(vidoefile)
        print(frame.shape)


    detection_graph = load_detec_graph(detec_model_path)
    label_map = load_detec_labe_map(detec_label_map_path)
    recog_graph = load_recog_graph(recog_model_path)
    recog_labels = load_recog_labels(recog_label_path)

    with recog_graph.as_default():
        input_img = tf.placeholder(tf.uint8, [None, None, 3], name='input_img')
        input_tensor = read_tensor_from_image_file(input_img,
                                                   input_height=input_height, input_width=input_width,
                                                   input_mean=input_mean, input_std=input_std)
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = recog_graph.get_operation_by_name(input_name)
        output_operation = recog_graph.get_operation_by_name(output_name)

    with detection_graph.as_default():
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        tensor_dict = inference_for_single_image(image_tensor, detection_graph)

    detec_sess = tf.Session(graph=detection_graph)
    recog_sess = tf.Session(graph=recog_graph)

    while ret:

        if isvideo == True:
            if loop_flag == pos:
                loop_flag = loop_flag + 1
                cv2.setTrackbarPos('time', 'gesture', loop_flag)
            else:
                pos = cv2.getTrackbarPos('time', 'gesture')
                loop_flag = pos
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

            if False:
                for ncountframe in range(0,1):
                    ret, frame = cap.read()
                    if ret == False:
                        return
                    loop_flag = loop_flag + 1
                    cv2.setTrackbarPos('time', 'gesture', loop_flag)

        org_img = frame.copy()
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)

        image_np = frame.copy()
        hand_img = None
        hand_str = None

        start = time.time()
        output_dict = detec_sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
        end = time.time()
        use_time = end-start
        print("Detection time: {:.3f}s".format(end - start))

        # output_dict['detection_scores'] = output_dict['detection_scores'][0]
        # 若检测到手
        # max_score = output_dict['detection_scores'][0]

        num_detections = int(output_dict['num_detections'][0])
        gestime = 0

        hand_strs = []
        hand_images = []

        for stepss in range(0, num_detections):

            if output_dict['detection_scores'][0][stepss] < 0.2:  #threadhold
                break
            if output_dict["detection_classes"][0][stepss] == 2 or output_dict["detection_classes"][0][stepss] == 3:
                box0 = output_dict['detection_boxes'][0][stepss]
                template = "{} (score={:0.5f})"
                display_str = template.format(label_map[output_dict["detection_classes"][0][stepss]]["name"], output_dict['detection_scores'][0][stepss])
                '''for i in top_k:
                    print(template.format(recog_labels[i], results[i]))
                print('')'''
                visualization(image_np, box0, display_str, thickness=8, use_normalized_coordinates=True,color="blue")
            if output_dict["detection_classes"][0][stepss] == 1:  # is hand
                classesid = output_dict['detection_classes'][0][stepss].astype(np.uint8)

                box0 = output_dict['detection_boxes'][0][stepss]
                sz = image_np.shape
                x = sz[0]
                y = sz[1]
                box0 = get_roi_box(box0,1.0)
                roi = frame[int(x * box0[0]): int(x * box0[2]), int(y * box0[1]): int(y * box0[3])]
                hand_img = roi.copy()
                hand_images.append(hand_img)
                feed = recog_sess.run(input_tensor, feed_dict={input_img: roi})
                start = time.time()
                results = recog_sess.run(output_operation.outputs[0],
                                         {input_operation.outputs[0]: feed})
                end = time.time()
                gestime += end-start
                results = np.squeeze(results)
                top_k = results.argsort()[-5:][::-1]
                print('Recognition time: {:.3f}s\n'.format(end - start))
                template = "{} (score={:0.3f})"
                if results[top_k[0]] > 0.01:
                    display_str = template.format(recog_labels[top_k[0]], results[top_k[0]])
                    hand_str = recog_labels[top_k[0]]
                    hand_strs.append(hand_str)
                    for i in top_k:
                        print(template.format(recog_labels[i], results[i]))
                    print('')
                    visualization(image_np, box0, display_str, thickness=8, use_normalized_coordinates=True)

        print("use the total time:%0.3f"%(use_time+gestime))
        cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB, image_np)

        cv2.imshow('gesture', image_np)
        key = cv2.waitKey(waitkeytime) & 0xFF
        if key == ord('s') :
            filename = time.strftime("%Y-%m-%d.%H-%M-%S", time.localtime())
            filename = "save/"+filename+".jpg"
            try:
                for count in range(len(hand_images)):
                    filename_hand = "save/" + time.strftime("%Y-%m-%d.%H-%M-%S_", time.localtime())+hand_strs[count] +"_"+str(count)+ "_hand.jpg"

                    hand_ori = hand_images[count]
                    cv2.cvtColor(hand_ori, cv2.COLOR_RGB2BGR, hand_ori)
                    cv2.imwrite(filename_hand,hand_ori)
            except:
                print("the hand image is error")
            #cv2.imwrite(filename, org_img)  #保存完整图片

        if key == 27 or key == ord('q') or isimage:
            cv2.destroyWindow('gesture')
            break


        ret, frame = cap.read()

    recog_sess.close()
    detec_sess.close()
    if cap != None:
        cap.release()


if __name__ == '__main__':
    detec_model_path = 'ssd_v1_picked_500x500.pb'
    detec_label_map_path = 'HAND_FACE_PHONE_label_map.pbtxt'


    recog_model_path = 'all_eight_graph.pb'
    recog_label_path = 'all_eight_labels.txt'





    recog_model_path = 'all_eight_graph.pb'
    recog_label_path = 'all_eight_labels.txt'

    recog_model_path = 'six_graph.pb'
    recog_label_path = 'six_labels.txt'
    
    recog_model_path = 'retrained_graph.pb'
    recog_label_path = 'retrained_labels.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--detec_model_path')
    parser.add_argument('--detec_label_map_path')
    parser.add_argument('--recog_model_path')
    parser.add_argument('--recog_label_path')
    args = parser.parse_args()

    if args.detec_model_path:
        detec_model_path = args.detec_model_path
    if args.detec_label_map_path:
        detec_label_map_path = args.detec_label_map_path
    if args.recog_model_path:
        recog_model_path = args.recog_model_path
    if args.recog_label_path:
        recog_label_path = args.recog_label_path
    vidoefile = "/Volumes/ss/hand/2018-12-2.mov"
    #vidoefile = "save/2018-12-01.17-47-40.jpg"
    main(vidoefile,1)
