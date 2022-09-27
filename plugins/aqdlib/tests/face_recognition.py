import aquila as aq
from argparse import ArgumentParser

import signal
import sys
import time

global loop
loop = True

def signal_handler(sig, frame):
    global loop
    print('You pressed Ctrl+C!')
    loop = False
signal.signal(signal.SIGINT, signal_handler)

parser = ArgumentParser()


parser.add_argument('--path', default="v4l2src device=/dev/video2 ! image/jpeg ! queue ! decodebin ! videoconvert ! queue ! appsink")
parser.add_argument('--cfg', default='/home/dan/code/yolo_tiny_face/yolov3-tiny.cfg')
parser.add_argument('--weights', default='/home/dan/code/yolo_tiny_face/yolov3-tiny_final.weights')
parser.add_argument('--labels', default='/home/dan/code/yolo_tiny_face/labels.txt')

args = parser.parse_args()

stream = aq.createStream(name='main')
aq.Stream.setCurrent(stream)
graph = aq.Graph(stream=stream)


fg = aq.framegrabbers.create(args.path)
assert fg is not None, 'Unable to load {}'.format(args.path)
#fg.logging_verbosity = 'trace'
graph.addNode(fg)
face = aq.nodes.YOLO(graph=graph, input=fg, name="face_detector", queue_size=1)
#face.logging_verbosity = 'trace'
face.weight_file = args.weights
face.model_file = args.cfg
face.label_file = args.labels
face.det_thresh = 0.01
face.cat_thresh = 0.01

aligner = aq.nodes.FaceAligner(image=fg,
                                detections=face,
                                shape_landmark_file='/home/dan/code/EagleEye/plugins/aqdlib/share/shape_predictor_5_face_landmarks.dat',
                                queue_size=300)

recognizer = aq.nodes.FaceRecognizer(image=fg, detections=aligner,
                                     queue_size=300,
                                     face_recognizer_weight_file='/home/dan/code/EagleEye/plugins/aqdlib/share/dlib_face_recognition_resnet_model_v1.dat')

facedb = aq.nodes.FaceDatabase(graph=graph, detections=recognizer, image=fg, queue_size=300)
facedb.unknown_detections = './unknown'
facedb.recent_detections = './recent'
facedb.known_detections ='./known'

draw = aq.nodes.DrawDetections(image=fg, detections=face, name="face_renderer", queue_size=300)
disp = aq.nodes.QtImageDisplay(input=draw, name="image_display")
start = time.time()

def printInputBufferSizes(obj):
    inputs = obj.getInputs()
    for x in inputs:
        print('{} buffer size = {}'.format(x.getName(), x.getInputBufferSize()))

while(loop and (time.time() - start) < 10.0):
    graph.step()
    stream.synchronize()

#    printInputBufferSizes(disp)
#    printInputBufferSizes(draw)
#    printInputBufferSizes(disp)
print('---------------------- Script exit ---------------------')

