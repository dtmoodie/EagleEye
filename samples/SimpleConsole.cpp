
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

    auto rootNode = manager.addNode("SerialStack");
    rootNode->addChild(manager.addNode("TestNode"));
    rootNode->addChild(manager.addNode("TestChildNode"));

    cv::cuda::GpuMat img;
    while(1)
    {
        manager.CheckRecompile();
        img = rootNode->process(img);
#if _WIN32
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
#else
        usleep(1000*1000);
#endif
    }
}
