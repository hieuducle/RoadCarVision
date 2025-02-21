import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
from torchvision import transforms
import torch
import json
from datasets import load_class_names, separate_class
from models import construct_model
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Car")
    parser.add_argument("--video-path", "-v", type=str, default="/home/amin/PycharmProjects/PythonProject/car2way.mp4")
    parser.add_argument("--img-size", "-s", type=int, default=400)
    parser.add_argument("--model-yolo", "-md", type=str, default="yolov8n.pt")
    parser.add_argument("--config-cls", "-c", type=str,
                        default="/home/amin/PycharmProjects/PythonProject/RoadCarVision/logs/resnext50_400_60_v3/1/config.json")
    parser.add_argument("--model-cls", "-mc", type=str,
                        default="/home/amin/PycharmProjects/PythonProject/RoadCarVision/logs/resnext50_400_60_v3/1/best.pth")
    args = parser.parse_args()
    return args


def draw_corner_box(frame, bbox, color=(255, 200, 100), thickness=3, corner_length=10):
    xmin, ymin, xmax, ymax = bbox

    cv2.line(frame, (xmin, ymin), (xmin + corner_length, ymin), color, thickness)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_length), color, thickness)

    cv2.line(frame, (xmax, ymin), (xmax - corner_length, ymin), color, thickness)
    cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_length), color, thickness)

    cv2.line(frame, (xmin, ymax), (xmin + corner_length, ymax), color, thickness)
    cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_length), color, thickness)

    cv2.line(frame, (xmax, ymax), (xmax - corner_length, ymax), color, thickness)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_length), color, thickness)

def load_model_cls():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = args.config_cls
    model_path = args.model_cls

    config = json.load(open(config_path))
    config['imgsize'] = (args.img_size, args.img_size)

    class_names = load_class_names()
    num_classes = len(class_names)
    v2_info = separate_class(class_names)
    make_names = v2_info['make'].unique()
    num_makes = len(make_names)
    model_type_names = v2_info['model_type'].unique()
    num_types = len(model_type_names)

    model = construct_model(config, num_classes, num_makes, num_types)
    # load_weight(model, model_path, device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()

    return model, class_names, make_names, model_type_names, device

def is_inside_roi(box, roi):
    xmin, ymin, xmax, ymax = map(int, box)
    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
    return cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0



def image_processing(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.fromarray(img.astype('uint8'))
    img = img.convert('RGB')
    img = transforms.Resize(args.img_size)(img)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    img = test_transform(img).float()
    img = img.to(device).unsqueeze(0)
    return img

def draw_roi(frame,roi,fill=True):
    frame = cv2.polylines(frame, [roi], isClosed=True, color=(0, 255, 0), thickness=1)

    if fill:
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(mask, [roi], (0, 255, 0))
        frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
    return frame


def cls_names(model, class_names, make_names, model_type_names, device, img):
    with torch.no_grad():
        pred, make_pred, model_type_pred = model(img)
        class_idx = pred.max(1)[1].item()
        cls = class_names[class_idx]

        make_idx = make_pred.max(1)[1].item()
        make = make_names[make_idx]

        model_type_idx = model_type_pred.max(1)[1].item()
        model_type = model_type_names[model_type_idx]

        return "Car Model: {}".format(cls)

def main(args):
    model = YOLO(args.model_yolo)
    tracker = DeepSort(max_age=30)
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # delay = int(1000 / fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter("output/222way.mp4", fourcc, fps, (1600, 880))
    line_y = 440
    cnt_1 = 0
    cnt_2 = 0
    crossed_car_1 = set()
    crossed_car_2 = set()
    roi1 = np.array([(405, 229), (611, 229), (768, 880), (0, 880), (0, 554)])
    roi2 = np.array([(663,229),(863,229),(1600,600),(1600,880),(1042,880)])
    roi3 = np.array([(144, line_y), (1285, line_y), (1508, 554), (0, 554)])

    track_classes = {}
    track_positions = {}
    car_model, class_names, make_names, model_type_names, device = load_model_cls()

    fixed_speeds = {}
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (1600, 880))
        frame = draw_roi(frame,roi1)
        frame = draw_roi(frame,roi2)
        frame = draw_roi(frame,roi3,False)

        results = model(frame)[0]
        detections = []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            cls_id = int(cls.item())
            class_name = model.names[cls_id]
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            if class_name == "car" and xmin > 240:

                img = frame[ymin:ymax, xmin:xmax]
                img = image_processing(img)
                text = cls_names(car_model, class_names, make_names, model_type_names, device, img)
                detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], conf.item(), text))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track, det in zip(tracks, detections):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_name = det[2]
            track_classes[track_id] = class_name

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            xmin, ymin, xmax, ymax = map(int, track.to_ltrb())
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

            class_name = track_classes.get(track_id, "Unknown")

            if track_id not in track_positions:
                track_positions[track_id] = []
            track_positions[track_id].append((center_x, center_y))

            speed_kmh = 0
            if len(track_positions[track_id]) > 2:
                x1, y1 = track_positions[track_id][-2]
                x2, y2 = track_positions[track_id][-1]

                pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                meter_distance = pixel_distance * 0.17
                time_seconds = 1 / fps
                speed_kmh = (meter_distance / time_seconds) * 3.6

            if ymin > line_y and track_id not in crossed_car_1 and is_inside_roi((xmin, ymin, xmax, ymax), roi1):
                cnt_1 += 1
                crossed_car_1.add(track_id)
            elif ymin < line_y and track_id not in crossed_car_2 and is_inside_roi((xmin, ymin, xmax, ymax), roi2):
                cnt_2 += 1
                crossed_car_2.add(track_id)

            if is_inside_roi((xmin, ymin, xmax, ymax), roi3):
                if track_id not in fixed_speeds:
                    fixed_speeds[track_id] = speed_kmh
                speed = "{:.2f} km/h".format(fixed_speeds[track_id])
                cv2.putText(frame, speed, (xmin, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if is_inside_roi([xmin,ymin,xmax,ymax],roi3):
                draw_corner_box(frame, [xmin, ymin, xmax, ymax],(0,0,255))
            else:

                draw_corner_box(frame,[xmin,ymin,xmax,ymax])
            text = "{}".format(class_name)
            cv2.putText(frame, text, (xmin - 75, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        cv2.putText(frame, "In {}".format(cnt_1), (160, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(frame, "Out {}".format(cnt_2), (1030, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.line(frame, (144, line_y), (1285, line_y), (0, 0, 255), 2)
        cv2.line(frame, (1508, 554), (0, 554), (0, 0, 255), 2)
        cv2.circle(frame,(190,175),75,(255, 200, 100),3)
        cv2.circle(frame, (1070, 175), 75, (255, 200, 100), 3)


        # out.write(frame)
        # cv2.imshow("Tracking", frame)
        # if cv2.waitKey(25) == ord("q"):
        #     break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    args = get_args()
    main(args)