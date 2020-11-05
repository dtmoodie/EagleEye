import aquila as aq
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument('--path', default='dog.jpg')
parser.add_argument('--cfg', default='/home/asiansensation/code/darknet/cfg/yolov3.cfg')
parser.add_argument('--weights', default='/home/asiansensation/code/darknet/cfg/yolov3.weights')

args = parser.parse_args()

stream = aq.createStream()
aq.setGuiStream(stream)
graph = aq.Graph()
graph.setStream(stream)

fg = aq.framegrabbers.create(args.path)
graph.addNode(fg)

detector = aq.nodes.YOLO(input=fg)

detector.weight_file = '/home/asiansensation/code/darknet/cfg/yolov3-tiny.weights'
detector.model_file = '/home/asiansensation/code/darknet/cfg/yolov3-tiny.cfg'
detector.label_file = '/home/asiansensation/code/darknet/data/coco.names'
#detector.channel_mean = [0,0,0,0]
#detector.pixel_scale = 1.0
detector.det_thresh = 0.2
detector.cat_thresh = 0.1

draw = aq.nodes.DrawDetections(image=fg, detections=detector)
writer = aq.nodes.ImageWriter(input_image=draw)
writer.request_write = True

print(detector)
print('Weight file: \n{}'.format(detector.weight_file))

aq.log('trace')
graph.start()

aq.eventLoop(10000)
output = detector.output

components = output.data.providers
for component in components:
    print(component.data)