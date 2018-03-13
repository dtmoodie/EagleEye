import aquila as aq
graph = aq.Graph()
fg = aq.framegrabbers.create('http://192.168.0.99:80')
flip = aq.nodes.Flip(input=fg)

face = aq.nodes.HaarFaceDetector(model_file='/home/dan/code/opencv/data/haarcascades/haarcascade_frontalface_default.xml', input=flip)
recognizer = aq.nodes.FaceRecognizer(image=flip, detections=face,
    shape_landmark_file='/home/dan/code/eagleeye/plugins/dlib/share/shape_predictor_5_face_landmarks.dat',
    face_recognizer_weight_file='/home/dan/code/eagleeye/plugins/dlib/share/dlib_face_recognition_resnet_model_v1.dat')
facedb = aq.nodes.FaceDatabase(detections=recognizer, image=flip)
draw = aq.nodes.DrawDescriptors(image=flip, detections=facedb)

disp = aq.nodes.OGLImageDisplay(image=draw)
graph.addNode(fg)
graph.start()
