import aquila as aq
from argparse import ArgumentParser

import signal
import sys


global loop
loop = True

def signal_handler(sig, frame):
    global loop
    print('You pressed Ctrl+C!')
    loop = False

signal.signal(signal.SIGINT, signal_handler)

parser = ArgumentParser()

parser.add_argument('--path', default="2")

args = parser.parse_args()

path ='v4l2src device=/dev/video{} ! image/jpeg ! decodebin ! videoconvert ! appsink'.format(args.path)

stream = aq.createStream()
#aq.setGuiStream(stream)

graph = aq.Graph()
graph.setStream(stream)

fg = aq.framegrabbers.create(path)
assert fg is not None, 'Unable to load {}'.format(path)
graph.addNode(fg)

disp = aq.nodes.QtImageDisplay(input=fg)

while(loop):
    graph.step()

