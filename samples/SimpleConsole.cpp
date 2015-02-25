
#include "EagleLib.h"
#include "Manager.h"
#include "nodes/Node.h"

int main()
{
    EagleLib::NodeManager manager;
    //std::vector<EagleLib::Node::Ptr> nodes;
	
	// Since manager might have been compiled in debug or release as opposed to this executable, we need to use the AUDynArray object
	// to pass the constructors into the manager, instead of the returned std::vector.
	ADD_CONSTRUCTORS(manager)

	auto node = manager.addNode("TestNode");
    node->updateParameter("Output", std::string("Parent!"));
    node->addParameter("Test", int(5));
    auto child = manager.addNode("TestChildNode");
	

    //node->updateParameter("Output",  std::string("Parent"));
    //nodes.push_back(node);
    //node = manager.addNode("TestNode");
    //node->updateParameter("Output", std::string("Child"));
    //nodes.push_back(node);
    //nodes[0]->addChild(nodes[1]);

    cv::cuda::GpuMat img;
    while(1)
    {
        manager.CheckRecompile();
        img = node->process(img);
		child->process(img);
#if _WIN32
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
#else
        usleep(1000*1000);
#endif
    }
}
