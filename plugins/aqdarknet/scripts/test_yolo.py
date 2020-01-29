import aquila as aq

graph = aq.Graph()


#fg = aq.framegrabbers.create('/home/asiansensation/code/darknet/data/dog.jpg')
fg = aq.framegrabbers.create('0')

yolo = aq.nodes.YOLO(input=fg)
yolo.set_label_file('@DARKNET_DIR@/data/coco.names')
yolo.set_model_file('@DARKNET_DIR@/cfg/yolov3-tiny.cfg')
yolo.set_weight_file('@DARKNET_DIR@/cfg/yolov3-tiny.weights')
yolo.set_swap_bgr(False)

draw = aq.nodes.DrawDetections(image=fg, detections=yolo)
fps = aq.nodes.FrameRate(input=draw, draw_fps=True)

pub = aq.nodes.ImagePublisher(input=fps)
graph.addNode(fg)

graph.start()
