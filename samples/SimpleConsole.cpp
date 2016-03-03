
#include <EagleLib/rcc/shared_ptr.hpp>
#include <EagleLib/nodes/NodeManager.h>
#include "EagleLib/nodes/Node.h"
#include "EagleLib/Plugins.h"
#include <EagleLib/DataStreamManager.h>
#include <EagleLib/Logging.h>
#include <signal.h>
#include <signals/logging.hpp>

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
#include <boost/tokenizer.hpp>

#include <parameters/Persistence/TextSerializer.hpp>

void PrintNodeTree(EagleLib::Nodes::Node* node, int depth)
{
    for(int i = 0; i < depth; ++i)
    {
        std::cout << "=";
    }
    std::cout << node->getFullTreeName() << std::endl;
    for(int i = 0; i < node->children.size(); ++i)
    {
        PrintNodeTree(node->children[i].get(), depth + 1);
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
	
    //boost::log::add_file_log(boost::log::keywords::file_name = "SimpleConsole%N.log", boost::log::keywords::rotation_size = 10 * 1024 * 1024);
    EagleLib::SetupLogging();
    
    
    desc.add_options()
        ("file", boost::program_options::value<std::string>(), "Required - File to load for processing")
        ("config", boost::program_options::value<std::string>(), "Required - File containing node structure")
        ("plugins", boost::program_options::value<boost::filesystem::path>(), "Path to additional plugins to load")
		("log", boost::program_options::value<std::string>()->default_value("info"), "Logging verbosity. trace, debug, info, warning, error, fatal")
        ("mode", boost::program_options::value<std::string>()->default_value("batch"), "Processing mode, options are interactive or batch")
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    
    if((!vm.count("config") || !vm.count("file")) && vm["mode"].as<std::string>() == "batch")
    {
        LOG(info) << "Batch mode selected but \"file\" and \"config\" options not set";
        std::cout << desc;
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
    {
    	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
    }
	boost::filesystem::path currentDir = boost::filesystem::current_path();
#ifdef _DEBUG
    currentDir.append("../Debug");
#else
    currentDir.append("../RelWithDebInfo");
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

    if(vm["mode"].as<std::string>() == "batch")
    {
        quit = false;
        std::string document = vm["file"].as<std::string>();
        LOG(info) << "Loading file: " << document;
        std::string configFile = vm["config"].as<std::string>();
        LOG(info) << "Loading config file " << configFile;
    

        auto stream = EagleLib::DataStreamManager::instance()->create_stream();
        stream->LoadDocument(document);
    
        auto nodes = EagleLib::NodeManager::getInstance().loadNodes(configFile);
        stream->AddNode(nodes);

        LOG(info) << "Loaded " << nodes.size() << " top level nodes";
        for(int i = 0; i < nodes.size(); ++i)
        {
            PrintNodeTree(nodes[i].get(), 1);
        }
        stream->process();
    }else
    {
        std::vector<std::shared_ptr<EagleLib::DataStream>> _dataStreams;
        EagleLib::DataStream* current_stream = nullptr;
        EagleLib::Nodes::Node* current_node = nullptr;

        auto print_options = []()->void
        {
            std::cout << 
                "- Options: \n "
                " - LoadFile {document}                       -- Create a frame grabber for a document \n"
                " - Add {node}                                -- Add a node to the current selected object\n"
                " - List {nodes}                              -- List all possible nodes that can be constructed\n"
                " - Print {streams,nodes,parameters, current} -- Prints the current streams, nodes in current stream, \n"
                "                                                or parameters of current node\n"
                " - Select {node,stream}                      -- Select a node by name (relative to current selection\n"
                "                                                or absolute, or a stream by index)\n"
                " - Save                                      -- Save node configuration\n"
                " - Load                                      -- Load node configuration\n"
                " - Help                                      -- Print this help\n"
                " - Quit                                      -- Close program and cleanup\n";
        };

        std::map<std::string, std::function<void(std::string)>> function_map;
        function_map["LoadFile"] = [&_dataStreams](std::string doc)->void
        {
            if(EagleLib::DataStream::CanLoadDocument(doc))
            {
                LOG(debug) << "Found a frame grabber which can load " << doc;
                auto stream = EagleLib::DataStreamManager::instance()->create_stream();
                if(stream->LoadDocument(doc))
                {
                    _dataStreams.push_back(stream);
                }else
                {
                    LOG(warning) << "Unable to load document";
                }
            }else
            {
                LOG(warning) << "Unable to find a frame grabber which can load " << doc;
            }
        };
        function_map["Quit"] = [](std::string)->void
        {
            quit = true;
        };
        function_map["Print"] = [&_dataStreams, &current_stream, &current_node](std::string what)->void
        {
            if(what == "streams")
            {
                for(auto& itr : _dataStreams)
                {
                    std::cout << " - " << itr->get_stream_id() << " - " << itr->GetFrameGrabber()->GetSourceFilename() << "\n";
                }
            }
            if(what == "nodes")
            {
                if(current_stream)
                {
                    auto nodes = current_stream->GetNodes();
                    for(auto& node : nodes)
                    {
                        PrintNodeTree(node.get(), 0);
                    }
                }
                else if(current_node)
                {
                    PrintNodeTree(current_node, 0);
                }
            }
            if(what == "parameters")
            {
                std::vector<std::shared_ptr<Parameters::Parameter>> parameters;
                if(current_node)
                {
                    parameters = current_node->getParameters();
                }
                if(current_stream)
                {
                    parameters = current_stream->GetFrameGrabber()->getParameters();
                }
                for(auto& itr : parameters)
                {
                    std::stringstream ss;
                    try
                    {
                        Parameters::Persistence::Text::Serialize(&ss, itr.get());   
                        std::cout << " - " << itr->GetTreeName() << ": " << ss.str() << "\n";
                    }catch(...)
                    {
                        std::cout << " - " << itr->GetTreeName() << "\n";
                    }
                }
            }
            if(what == "current")
            {
                if(current_stream)
                {
                    std::cout << " - Current stream: " << current_stream->GetFrameGrabber()->GetSourceFilename() << "\n";
                    return;
                }
                if(current_node)
                {
                    std::cout << " - Current node: " << current_node->getFullTreeName() << "\n";
                    return;
                }
                std::cout << "Nothing currently selected\n";
            }
        };
        function_map["Select"] = [&_dataStreams,&current_stream, &current_node](std::string what)
        {
            int idx = -1;
            std::string name;
            
            try
            {
                idx =  boost::lexical_cast<int>(what);
            }catch(...)
            {
                idx = -1;
            }
            if(idx == -1)
            {
                name = what;
            }
            if(idx != -1)
            {
                LOG(info) << "Selecting stream " << idx;
                for(auto& itr : _dataStreams)
                {
                    if(itr->get_stream_id() == idx)
                    {
                        current_stream = itr.get();
                        current_node = nullptr;
                        return;
                    }
                }
            }
            if(current_stream)
            {
                // look for a node with this name, relative then look absolute
                auto nodes = current_stream->GetNodes();
                for(auto& node : nodes)
                {
                    if(node->getTreeName() == what)
                    {
                        current_stream = nullptr;
                        current_node = node.get();
                        return;
                    }
                }
                // parse name and try to find absolute path to node

            }
          
        };
        function_map["Help"] = [&print_options](std::string)->void{print_options();};
        function_map["List"] = [](std::string)->void
        {
            auto nodes = EagleLib::NodeManager::getInstance().getConstructableNodes();
            for(auto& node : nodes)
            {
                std::cout << " - " << node << "\n";
            }
        };
        function_map["Add"] = [&current_node, &current_stream](std::string name)->void
        {
            auto node = EagleLib::NodeManager::getInstance().addNode(name);
            if(!node)
            {
                return;
            }
            if(current_node)
            {
                current_node->addChild(node);
                return;
            }
            if(current_stream)
            {
                current_stream->AddNode(node);
                return;
            }
        };
        
        print_options();
        
        while(!quit)
        {
            std::string command_line;
            std::getline(std::cin, command_line);
            int count = 0;
	        std::stringstream ss;
            ss << command_line;
            std::string command;
            std::getline(ss, command, ' ');
            if(function_map.count(command))
            {
                std::string rest;
                ss >> rest;
                LOG(debug) << "Executing command (" << command << ") with arguments: " << rest;
                function_map[command](rest);
            }else
            {
                LOG(warning) << "Invalid command: " << command_line;
                print_options();
            }
        }
        LOG(info) << "Shutting down";
    }
    
    
    return 0;
}
