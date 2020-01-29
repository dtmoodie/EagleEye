import aquila
graph = aquila.Graph()
fg = aquila.framegrabbers.create('@FRAMEGRABBERS_DATA_DIR@')
graph.addNode(fg)
img_disp = aquila.nodes.OGLImageDisplay(name='image', image=fg)
graph.step()
#graph.start()
