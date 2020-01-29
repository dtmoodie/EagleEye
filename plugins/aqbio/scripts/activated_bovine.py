import aquila as aq
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_folder', type=str)
args = parser.parse_args()

graph = aq.Graph()

fg = aq.framegrabbers.create(args.image_folder)
#fg.set_synchronous(True)

graph.addNode(fg)
gray = aq.nodes.ConvertToGrey(input=fg)
blur = aq.nodes.GaussianBlur(input=gray, sigma=2)
circles = aq.nodes.HoughCircle(input=blur, center_threshold=12)
membrane = aq.nodes.FindCellMembrane(input=blur, circles=circles)
membrane.set_alpha(2.0)
membrane.set_beta(2.0)
membrane.set_num_samples(400)
membrane.set_window_size(9)
membrane.set_inner_pad(1.1)
membrane.set_outer_pad(2.5)
membrane.set_radial_resolution(0.5)
membrane.set_radial_weight(4.5)
measure = aq.nodes.MeasureCell(cell=membrane, image_name=fg, image=fg, out_dir = args.image_folder + '_results')
draw = aq.nodes.DrawContours(input_image=fg, input_contours=membrane, draw_mode='All')
disp = aq.nodes.QtImageDisplay(image=draw)

graph.start()

graph.wait('eos')
print('done!')
