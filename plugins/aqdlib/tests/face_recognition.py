import aquila as aq

graph = aq.Graph()

devices = aq.framegrabbers.listDataSources()

cam = aq.framegrabbers.FrameGrabber()
graph.addNode(cam)

cam.loadData('v4l2src ! video/x-raw,width=1920,height=1080,format=RGB ! videoconvert ! appsink name=mysink')
face = aq.nodes.HaarFaceDetector(graph=graph, model_file='/home/dan/code/opencv/data/haarcascades/haarcascade_frontalface_default.xml', input=cam)
recognizer = aq.nodes.FaceRecognizer(image=cam, detections=face,
    shape_landmark_file='/code/eagleeye/plugins/dlib/share/shape_predictor_5_face_landmarks.dat',
    face_recognizer_weight_file='/code/eagleeye/plugins/dlib/share/dlib_face_recognition_resnet_model_v1.dat')
facedb = aquila.nodes.FaceDatabase(detections=recognizer, image=cam)
draw = aquila.nodes.DrawDescriptors(image=cam, detections=facedb)
img_disp = aquila.nodes.QtImageDisplay(name='image', image=draw)
#graph.start()
