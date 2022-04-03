import aquila as aq
from argparse import ArgumentParser
import time

parser = ArgumentParser()

parser.add_argument('--path', default='images')
parser.add_argument('--cfg', default='/home/dan/code/yolo_tiny_face/yolov3-tiny.cfg')
parser.add_argument('--weights', default='/home/dan/code/yolo_tiny_face/yolov3-tiny_final.weights')
parser.add_argument('--labels', default='/home/dan/code/yolo_tiny_face/labels.txt')

args = parser.parse_args()

stream = aq.createStream()
gui_stream = aq.getGUIStream();
graph = aq.Graph()
graph.setStream(stream)

fg = aq.framegrabbers.create(args.path)
graph.addNode(fg)
aq.log('warning')
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

facedb.unknown_detections = './unknown'
facedb.recent_detections = './recent'
facedb.known_detections ='./known'


draw = aq.nodes.DrawDetections(image=fg, detections=facedb)

display = aq.nodes.QtImageDisplay(input=draw)
writer = aq.nodes.ImageWriter(input_image=draw, request_write=True, frequency=1, save_directory='./overlays')

count = 0
while(not fg.getParam('eos').data.data):
    graph.step()
    #aq.eventLoop(10)
    gui_stream.synchronize()
    output = facedb.output

    components = output.data.components
    for component in components:
        print(component.data.data.typename)
        print(component.data.data.data)
    count += 1

facedb.saveUnknownFaces()
time.sleep(1)
