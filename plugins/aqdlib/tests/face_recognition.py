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


parser.add_argument('--path', default="v4l2src device=/dev/video2 ! image/jpeg ! queue ! decodebin ! videoconvert ! queue ! appsink")
parser.add_argument('--cfg', default='/home/dan/code/yolo_tiny_face/yolov3-tiny.cfg')
parser.add_argument('--weights', default='/home/dan/code/yolo_tiny_face/yolov3-tiny_final.weights')
parser.add_argument('--labels', default='/home/dan/code/yolo_tiny_face/labels.txt')

args = parser.parse_args()

stream = aq.createStream()
#aq.setGuiStream(stream)

graph = aq.Graph()
graph.setStream(stream)

fg = aq.framegrabbers.create(args.path)
assert fg is not None, 'Unable to load {}'.format(args.path)
graph.addNode(fg)

face = aq.nodes.YOLO(graph=graph, input=fg, queue_size=0.01)

face.weight_file = args.weights
face.model_file = args.cfg
face.label_file = args.labels
face.det_thresh = 0.01
face.cat_thresh = 0.01

aligner = aq.nodes.FaceAligner(graph=graph, image=fg, detections=face, shape_landmark_file='/home/dan/code/EagleEye/plugins/aqdlib/share/shape_predictor_5_face_landmarks.dat')

recognizer = aq.nodes.FaceRecognizer(graph=graph, image=fg, detections=aligner,
    face_recognizer_weight_file='/home/dan/code/EagleEye/plugins/aqdlib/share/dlib_face_recognition_resnet_model_v1.dat')

facedb = aq.nodes.FaceDatabase(graph=graph, detections=recognizer, image=fg)
facedb.unknown_detections = './unknown'
facedb.recent_detections = './recent'
facedb.known_detections ='./known'

draw = aq.nodes.DrawDetections(graph=graph, image=fg, detections=facedb)
disp = aq.nodes.QtImageDisplay(graph=graph, input=draw)
#writer = aq.nodes.ImageWriter(input_image=draw, request_write=True, frequency=1, save_directory='./')

while(loop):
    graph.step()
    #X = face.getInputs()
    #for x in X:
        #pub = x.getPublisher()
        #if pub is not None:
            #headers = pub.getAvailableHeaders()
            #print(len(headers))

