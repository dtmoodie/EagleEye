
#include "EagleLib.h"
#include "Manager.h"
#include "nodes/Node.h"

int main()
{
    EagleLib::NodeManager manager;
    //std::vector<EagleLib::Node::Ptr> nodes;
    auto node = manager.addNode("TestNode");
    node->updateParameter("Output", std::string("Parent!"));
    node->addParameter("Test", int(5));

    node->addChild(manager.addNode("TestChildNode"));

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
        node->process(img);
        usleep(1000*1000);
    }
}
