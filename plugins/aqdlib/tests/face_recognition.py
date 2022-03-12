import aquila as aq
from argparse import ArgumentParser

import signal
import sys


global loop
loop = True

def signal_handler(sig, frame):
    global loop
    print('You pressed Ctrl+C!')
    loop = False
signal.signal(signal.SIGINT, signal_handler)

parser = ArgumentParser()

parser.add_argument('--path', default='0')
parser.add_argument('--cfg', default='/home/dan/code/yolo_tiny_face/yolov3-tiny.cfg')
parser.add_argument('--weights', default='/home/dan/code/yolo_tiny_face/yolov3-tiny_final.weights')
parser.add_argument('--labels', default='/home/dan/code/yolo_tiny_face/labels.txt')

args = parser.parse_args()

stream = aq.createStream()
aq.setGuiStream(stream)

graph = aq.Graph()
graph.setStream(stream)

fg = aq.framegrabbers.create(args.path)
assert fg is not None, 'Unable to load {}'.format(args.path)
graph.addNode(fg)

face = aq.nodes.YOLO(input=fg)

face.weight_file = args.weights
face.model_file = args.cfg
face.label_file = args.labels
face.det_thresh = 0.01
face.cat_thresh = 0.01

aligner = aq.nodes.FaceAligner(image=fg, detections=face, shape_landmark_file='/home/dan/code/EagleEye/plugins/aqdlib/share/shape_predictor_5_face_landmarks.dat')

recognizer = aq.nodes.FaceRecognizer(image=fg, detections=aligner,
    face_recognizer_weight_file='/home/dan/code/EagleEye/plugins/aqdlib/share/dlib_face_recognition_resnet_model_v1.dat')

facedb = aq.nodes.FaceDatabase(detections=recognizer, image=fg)

draw = aq.nodes.DrawDetections(image=fg, detections=facedb)
disp = aq.nodes.QtImageDisplay(input=draw)
#writer = aq.nodes.ImageWriter(input_image=draw, request_write=True, frequency=1, save_directory='./')

for _ in range(100):
    graph.step()
    aq.eventLoop(10)
