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

#detector = aq.nodes.YOLO()

#aq.eventLoop(100000)