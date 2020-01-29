import aquila as aq
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--profile', type=bool, default=False)
parsed = parser.parse_args()

graph = aq.Graph()
graph.stop()

fg = aq.framegrabbers.create('0')
fg.getParam('height').data= 1080
fg.getParam('width').data= 1920
graph.addNode(fg)

yolo = aq.nodes.MXNet(input=fg)
yolo.set_network_width(256)
yolo.set_network_height(256)
yolo.set_pixel_scale(1.0)
yolo.set_image_scale(-1)
yolo.set_label_file('../../data/facedet_cuda/labels.txt')
yolo.set_model_file('../../data/yolo2.4/facedet-symbol.json')
yolo.set_weight_file('../../data/yolo2.4/facedet-0014.params')
yolo.getParam('iou_threshold').data = 0.1
yolo.getParam('detection_threshold').data = [0.3]

face = yolo

recognizer = aq.nodes.FaceRecognizer(image=fg, detections=face,
    shape_landmark_file='/home/asiansensation/code/eagleeye/plugins/aqdlib/share/shape_predictor_5_face_landmarks.dat',
    face_recognizer_weight_file='/home/asiansensation/code/eagleeye/plugins/aqdlib/share/dlib_face_recognition_resnet_model_v1.dat')
facedb = aq.nodes.FaceDatabase(detections=recognizer, image=fg)
facedb.set_database_path('/home/asiansensation/data/facedb/')

draw = aq.nodes.DrawDescriptors(image=fg, detections=facedb)
fps = aq.nodes.FrameRate(input=draw, draw_fps=True)

pub = aq.nodes.ImagePublisher(input=fps)

graph.start()

if(parsed.profile):
    time.sleep(10)
    graph.stop()

    del pub
    del fps
    del draw
    del facedb
    del recognizer
    del face
    del yolo
    del fg
    del graph

    print('Exiting')
