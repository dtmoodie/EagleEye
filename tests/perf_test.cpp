#include <EagleLib/nodes/NodeManager.h>
#include "EagleLib/nodes/Node.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

EagleLib::Nodes::Node::Ptr setVideoFile(EagleLib::Nodes::Node::Ptr node, const std::string& videoFile)
{
    if(node->getName() == "VideoLoader")
    {
        node->updateParameter<boost::filesystem::path>("Filename", boost::filesystem::path(videoFile));
        node->updateParameter<bool>("Loop", false);
        return node;
    }
    for(size_t i = 0; i < node->children.size(); ++i)
    {
        auto retNode = setVideoFile(node->children[i], videoFile);
        if(retNode != nullptr)
        {
            return retNode;
        }
    }
    return EagleLib::Nodes::Node::Ptr();
}

int main(int argc, char* argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
            ("Help", "Produce help message")
            ("nodeFile", po::value<std::string>(), "Set input file describing node map")
            ("videoFile",po::value<std::string>(), "Set file to process")
            ("loop", po::value<bool>(), "flag to loop video, defaults to false");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help") || vm.count("nodeFile") == 0)
    {
        std::cout << desc << std::endl;
        return 1;
    }

    auto nodes = EagleLib::NodeManager::getInstance().loadNodes(vm["nodeFile"].as<std::string>());
    EagleLib::NodeManager::getInstance().printNodeTree();
    EagleLib::Nodes::Node::Ptr playbackNode;
    bool loop = false;
    if(vm.count("loop"))
    {
        loop = vm["loop"].as<bool>();
    }
    if(vm.count("videoFile"))
    {
        std::string fileName = vm["videoFile"].as<std::string>();
        for(size_t i = 0; i < nodes.size(); ++i)
        {
           EagleLib::Nodes::Node::Ptr tmpNode = setVideoFile(nodes[i], fileName);
           if(tmpNode != nullptr)
           {
               playbackNode = tmpNode;
               playbackNode->updateParameter("Loop", loop);
           }
        }
        if(playbackNode == nullptr)
        {
            std::cout << "Error, no video playback node defined" << std::endl;
            return 1;
        }




        std::vector<cv::cuda::GpuMat> images(nodes.size());
        std::vector<cv::cuda::Stream> streams(nodes.size());
        while(!(*playbackNode->getParameter<bool>("End of video")->Data()))
        {
            for(size_t i = 0; i < nodes.size(); ++i)
            {
                images[i] = nodes[i]->process(images[i], streams[i]);
            }
        }
    }
    return 0;
}
