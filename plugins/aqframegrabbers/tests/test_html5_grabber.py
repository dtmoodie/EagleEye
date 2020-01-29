import aquila as aq
graph = aq.Graph()
fg = aq.framegrabbers.create('http://192.168.0.99:80')
disp = aq.nodes.OGLImageDisplay(image=fg)
graph.addNode(fg)
graph.start()
