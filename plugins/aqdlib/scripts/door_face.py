import aquila as aq
from argparse import ArgumentParser
import json
import socket
hostname = socket.gethostname()

flip_hosts = ['frontdoor']
if(hostname in flip_hosts):
    flip=True
else:
    flip=False

parser = ArgumentParser()
parser.add_argument('--haar_dir', type=str, default='/home/asiansensation/code/opencv/data/haarcascades')
parser.add_argument('--face_landmark_dir', type=str, default='@CMAKE_CURRENT_LIST_DIR@/share/')
parser.add_argument('--identities', type=str, default='/home/asiansensation/data/facedb')
parser.add_argument('--headless', action='store_false', default=True)
parser.add_argument('--flip', action='store_true', default=flip)
parser.add_argument('--source', type=str, default='0')
parser.add_argument('--detector', type=str, default='BrodmannFaceDetector')
parser.add_argument('--mqtt_broker', type=str, default='')
parser.add_argument('--model_dir', type=str, default='/home/asiansensation/data/yolo_tiny_face')
parser.add_argument('--dev', action='store_true', default=False)

parsed, unknown = parser.parse_known_args()
aq.readArgs(unknown)

width = 1280
height = 720

async = False

graph = aq.Graph()
#fg = aq.framegrabbers.create('http://192.168.0.99:80')
fg = aq.framegrabbers.create(parsed.source)

fg.getParam('height').data= height
fg.getParam('width').data= width

graph.addNode(fg)

if(parsed.flip):
    fg = aq.nodes.Flip(input=fg, axis='X')
    fg.set_roi(aq.datatypes.Rect2f(x=0.2,y=0.2, width=0.6, height=0.8))
else:
    fg = aq.nodes.Crop(input=fg)
    fg.set_roi(aq.datatypes.Rect2f(x=0.0,y=0.0, width=0.6, height=1.0))

if(parsed.detector == 'YOLO'):
    if(async):
        detection_graph = aq.nodes.SubGraph(graph=graph, name='detection')
    else:
        detection_graph=graph
    face = aq.nodes.YOLO(graph=detection_graph, input=fg)
    face.set_label_file(parsed.model_dir + '/labels.txt')
    face.set_model_file(parsed.model_dir + '/yolov3-tiny.cfg')
    face.set_weight_file(parsed.model_dir + '/yolov3-tiny_final.weights')
    face.set_swap_bgr(False)
    face.set_cat_thresh(0.01)
    face.set_det_thresh(0.01)
else:
    detectors = aq.nodes.FaceDetector.list()

    if(not parsed.detector in detectors):
        parsed.detector = detectors[0]

    if(async):
        detection_graph = aq.nodes.SubGraph(graph=graph, name='detection')
        limiter = aq.nodes.FrameLimiter(input=fg, graph=detection_graph, desired_framerate=1)
    else:
        detection_graph=graph
        limiter = fg
    cmd = 'face = aq.nodes.{}(input=limiter,graph=detection_graph)'.format(parsed.detector)
    exec(cmd)


if(async):
    #recognition_graph = aq.nodes.SubGraph(graph=graph, name='recognition')
    recognition_graph = detection_graph
    tracker = aq.nodes.DlibCorrelationTracker(graph=recognition_graph, image=fg, detections=face)
    tracker.setSyncParam("image")
else:
    recognition_graph = graph
    tracker = face



aligner = aq.nodes.FaceAligner(graph=recognition_graph, image=fg, detections=tracker,
    shape_landmark_file=parsed.face_landmark_dir + '/shape_predictor_5_face_landmarks.dat')

recognizer = aq.nodes.FaceRecognizer(graph=recognition_graph, image=fg, detections=aligner,
    face_recognizer_weight_file=parsed.face_landmark_dir + '/dlib_face_recognition_resnet_model_v1.dat')


facedb = aq.nodes.FaceDatabase(graph=recognition_graph, detections=recognizer, image=fg)
facedb.set_database_path(parsed.identities)
facedb.set_unknown_detections(parsed.identities)
facedb.set_known_detections(parsed.identities)
facedb.set_recent_detections(parsed.identities)
facedb.set_min_distance(0.45)

draw = aq.nodes.DrawDetections(graph=recognition_graph, image=fg, detections=facedb)

fps = aq.nodes.FrameRate(graph=recognition_graph, input=draw, draw_fps=True)

pub = aq.nodes.ImagePublisher(graph=recognition_graph, input=fps)

if(parsed.mqtt_broker):
    import paho.mqtt.client as mqtt
    dets = facedb.getOutputs()[0]
    client = mqtt.Client('eagleeye')
    client.connect(parsed.mqtt_broker)

    def callback(data, ts, fn, flags):
        try:
            if(len(data)):
                obj = []
                for det in data:
                    if(not 'unknown' in det.classifications[0].cat.name):
                        obj.append({
                            'ATTR_CONFIDENCE': int(det.classifications[0].conf*100),
                            'ATTR_NAME': det.classifications[0].cat.name,
                            'ATTR_LAST_SEEN': ts
                            })
                if(len(obj)):
                    client.publish('home-assistant/faces', json.dumps(obj))
        #except IOError e:
        #    print('sigpipe :/')
        except:
            print('Failed to publish detections')

    dets.setCallback(callback)

#aq.log('trace')
graph.start()

def cleanup():
    if(async):
        del pub
        del filter
        del fps
        del overlay
        del draw
        del facedb
        del recognizer
        del aligner
        del tracker
        del recognition_graph
        del face
        del limiter
        del detection_graph
