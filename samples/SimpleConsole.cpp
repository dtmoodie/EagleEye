
#include <EagleLib/shared_ptr.hpp>
#include <EagleLib/NodeManager.h>
#include "nodes/Node.h"
#include "Plugins.h"
#include <EagleLib/DataStreamManager.h>

#include <signal.h>

#include <boost/program_options.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/exceptions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/filesystem.hpp>
#include <boost/version.hpp>




void PrintNodeTree(EagleLib::Nodes::Node::Ptr node, int depth)
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
    //std::cout << "Caught signal " << s << std::endl;
	BOOST_LOG_TRIVIAL(error) << "Caught signal " << s;
    quit = true;
}

int main(int argc, char* argv[])
{
	signal(SIGINT, sig_handler);
	signal(SIGILL, sig_handler);
	signal(SIGTERM, sig_handler);
    boost::program_options::options_description desc("Allowed options");
	boost::log::add_file_log(boost::log::keywords::file_name = "SimpleConsole%N.log", boost::log::keywords::rotation_size = 10 * 1024 * 1024);
    desc.add_options()
        ("file", boost::program_options::value<std::string>(), "Required - File to load for processing")
        ("config", boost::program_options::value<std::string>(), "Required - File containing node structure")
        ("plugins", boost::program_options::value<boost::filesystem::path>(), "Path to additional plugins to load")
		("log", boost::program_options::value<std::string>(), "Logging verbosity. trace, debug, info, warning, error, fatal")
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    
    if(!vm.count("config") || !vm.count("file"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
	if (vm.count("log"))
	{
		std::string verbosity = vm["log"].as<std::string>();
		if (verbosity == "trace")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::trace);
		if (verbosity == "debug")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
		if (verbosity == "info")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
		if (verbosity == "warning")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
		if (verbosity == "error")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::error);
		if (verbosity == "fatal")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::fatal);
	}else
		boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
#ifdef _DEBUG
    boost::filesystem::path currentDir("./Debug");
#else
    boost::filesystem::path currentDir("./RelWithDebInfo");
#endif
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
    if(vm.count("plugins"))
    {
        currentDir = boost::filesystem::path(vm["plugins"].as<boost::filesystem::path>());
        for (boost::filesystem::directory_iterator itr(currentDir); itr != end_itr; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->path()))
            {
#ifdef _MSC_VER
                if (itr->path().extension() == ".dll")
#else
                if (itr->path().extension() == ".so")
#endif
                {
                    std::string file = itr->path().string();
                    EagleLib::loadPlugin(file);
                }
                else
                {
                    std::cout << itr->path().extension() << std::endl;
                }
            }
        }
    }
    quit = false;
    std::string document = vm["file"].as<std::string>();
    std::cout << "Loading file: " << document << std::endl;
    std::string configFile = vm["config"].as<std::string>();
    std::cout << "Loading config file " << configFile << std::endl;
    

    auto stream = EagleLib::DataStreamManager::instance()->create_stream();
    stream->LoadDocument(document);
    auto nodes = EagleLib::NodeManager::getInstance().loadNodes(configFile);
    stream->AddNode(nodes);
    stream->process();



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
        //EagleLib::ProcessingThreadCallback::Run();
        for(int i = 0; i < nodes.size(); ++i)
        {
            nodes[i]->process(images[i], streams[i]);
        }
    }
}
