import aquila as aq
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument('--path', default='dog.jpg')
parser.add_argument('--cfg', default='/home/dan/code/ml/darknet/cfg/yolov4.cfg')
parser.add_argument('--weights', default='/home/dan/code/ml/darknet/cfg/yolov4.weights')
parser.add_argument('--labels', default='/home/dan/code/ml/darknet/data/coco.names')

args = parser.parse_args()

stream = aq.createStream()
aq.setGuiStream(stream)
graph = aq.Graph()
graph.setStream(stream)

fg = aq.framegrabbers.create(args.path)
graph.addNode(fg)

detector = aq.nodes.YOLO(input=fg)

#detector.pixel_scale=1.0
detector.channel_mean = [0.0, 0.0, 0.0, 0.0]
detector.weight_file = args.weights
detector.model_file = args.cfg
detector.label_file = args.labels
detector.det_thresh = 0.8
detector.cat_thresh = 0.5

draw = aq.nodes.DrawDetections(image=fg, detections=detector)
writer = aq.nodes.ImageWriter(input_image=draw)
writer.request_write = True

print(detector)
print('Weight file: \n{}'.format(detector.weight_file))

aq.log('trace')

graph.step()
output = detector.output

components = output.data.components
for component in components:
    print(component.type)
    print(component.data.data.data)