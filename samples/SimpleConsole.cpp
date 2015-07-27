
#include "EagleLib.h"
#include "Manager.h"
#include "nodes/Node.h"
#include "Plugins.h"
#include <boost/program_options.hpp>
#include <signal.h>
//#include <unistd.h>


void PrintNodeTree(EagleLib::Node::Ptr node, int depth)
{
    for(int i = 0; i < depth; ++i)
    {
        std::cout << "=";
    }
    std::cout << node->fullTreeName << std::endl;
    for(int i = 0; i < node->children.size(); ++i)
    {
        PrintNodeTree(node->children[i], depth + 1);
    }
}
static volatile bool quit;
void sig_handler(int s)
{
    std::cout << "Cought signal " << s << std::endl;
    quit = true;
}

int main(int argc, char* argv[])
{
	signal(SIGINT, sig_handler);
	/*
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = sig_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
	*/



    boost::program_options::options_description desc("Allowed options");

    desc.add_options()
        ("config", boost::program_options::value<std::string>(), "File containing node structure")
        ("plugins", boost::program_options::value<std::string>(), "Path to additional plugins to load")
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

    if(!vm.count("config"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    boost::filesystem::path currentDir(".");
    boost::filesystem::directory_iterator end_itr;

    for(boost::filesystem::directory_iterator itr(currentDir); itr != end_itr; ++itr)
    {
        if(boost::filesystem::is_regular_file(itr->path()))
        {
#ifdef _MSC_VER
            if(itr->path().extension() == ".dll")
#else
            if(itr->path().extension() == ".so")
#endif
            {
                std::string file = itr->path().string();
                EagleLib::loadPlugin(file);
            }else
            {
                std::cout << itr->path().extension() << std::endl;
            }
        }
    }
    quit = false;
    std::string configFile = vm["config"].as<std::string>();
    std::cout << "Loading config file " << configFile << std::endl;
    auto nodes = EagleLib::NodeManager::getInstance().loadNodes(configFile);
    std::cout << "Loaded " << nodes.size() << " top level nodes" << std::endl;
    for(int i = 0; i < nodes.size(); ++i)
    {
        PrintNodeTree(nodes[i], 1);
    }
    // Start processing loop
    std::vector<cv::cuda::GpuMat> images;
    std::vector<cv::cuda::Stream> streams;
    images.resize(nodes.size());
    streams.resize(nodes.size());
    while(!quit && nodes.size())
    {
        EagleLib::ProcessingThreadCallback::Run();
        for(int i = 0; i < nodes.size(); ++i)
        {
            nodes[i]->process(images[i], streams[i]);
        }
    }
}
