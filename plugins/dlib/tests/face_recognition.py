import aquila

graph = aquila.Graph()

devices = aquila.framegrabbers.listDataSources()

cam = aquila.framegrabbers.FrameGrabber()
graph.addNode(cam)

cam.loadData('0')
face = aquila.nodes.HaarFaceDetector(model_file='/home/dan/code/opencv/data/haarcascades/haarcascade_frontalface_default.xml', input=cam)
recognizer = aquila.nodes.FaceRecognizer(image=cam, detections=face,
    shape_landmark_file='/home/dan/code/eagleeye/plugins/dlib/share/shape_predictor_5_face_landmarks.dat',
    face_recognizer_weight_file='/home/dan/code/eagleeye/plugins/dlib/share/dlib_face_recognition_resnet_model_v1.dat')
facedb = aquila.nodes.FaceDatabase(detections=recognizer, image=cam)
draw = aquila.nodes.DrawDescriptors(image=cam, detections=facedb)
img_disp = aquila.nodes.OGLImageDisplay(name='image', image=draw)
graph.start()
