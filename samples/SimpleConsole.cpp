
#include "EagleLib.h"
#include "Manager.h"
#include "nodes/Node.h"

int main()
{
    EagleLib::NodeManager& manager = EagleLib::NodeManager::getInstance();
    //std::vector<EagleLib::Node::Ptr> nodes;
	
	// Since manager might have been compiled in debug or release as opposed to this executable, we need to use the AUDynArray object
	// to pass the constructors into the manager, instead of the returned std::vector.
	manager.setupModule(PerModuleInterface::GetInstance());

    auto rootNode = manager.addNode("SerialStack");
	auto child = rootNode->addChild(manager.addNode("TestNode"));
    auto inputNode = rootNode->addChild(manager.addNode("TestChildNode"));
//	auto list = child->listParameters();
//	auto test = manager.getNode(child->fullTreeName);
	auto inputs = inputNode->findCompatibleInputs();
    for (size_t i = 0; i < inputs.size(); ++i)
	{
		if (inputs[i].size())
			inputNode->setInputParameter(inputs[i][0], i);
	}
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
