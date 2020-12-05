import aquila as aq
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--path', default='images')
parser.add_argument('--cfg', default='/home/asiansensation/code/yolo_tiny_face/yolov3-tiny.cfg')
parser.add_argument('--weights', default='/home/asiansensation/code/yolo_tiny_face/yolov3-tiny_final.weights')
parser.add_argument('--labels', default='/home/asiansensation/code/yolo_tiny_face/labels.txt')

args = parser.parse_args()

stream = aq.createStream()
aq.setGuiStream(stream)

graph = aq.Graph()
graph.setStream(stream)

fg = aq.framegrabbers.create(args.path)
graph.addNode(fg)

face = aq.nodes.YOLO(input=fg)

face.weight_file = args.weights
face.model_file = args.cfg
face.label_file = args.labels
face.det_thresh = 0.01
face.cat_thresh = 0.01

aligner = aq.nodes.FaceAligner(image=fg, detections=face, shape_landmark_file='/home/asiansensation/code/EagleEye/plugins/aqdlib/share/shape_predictor_5_face_landmarks.dat')

recognizer = aq.nodes.FaceRecognizer(image=fg, detections=aligner,
    face_recognizer_weight_file='/home/asiansensation/code/EagleEye/plugins/aqdlib/share/dlib_face_recognition_resnet_model_v1.dat')

facedb = aq.nodes.FaceDatabase(detections=recognizer, image=fg)

draw = aq.nodes.DrawDetections(image=fg, detections=facedb)

writer = aq.nodes.ImageWriter(input_image=draw)
writer.request_write = True
writer.frequency = 1

graph.start()
aq.eventLoop(10000)

output = facedb.output

components = output.data.components
for component in components:
    print(component.data.data.typename)
    print(component.data.data.data)
    if('Classification' in component.data.data.typename):
        assert component.data.data.data[0].data[0].cat.data.name == 'JerryRyan'


