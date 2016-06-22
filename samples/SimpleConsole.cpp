
#include <EagleLib/rcc/shared_ptr.hpp>
#include <EagleLib/nodes/NodeManager.h>
#include "EagleLib/nodes/Node.h"
#include "EagleLib/Plugins.h"
#include <EagleLib/DataStreamManager.h>
#include <EagleLib/Logging.h>
#include <EagleLib/rcc/ObjectManager.h>
#include <EagleLib/frame_grabber_base.h>
#include <EagleLib/DataStreamManager.h>

#include <signal.h>
#include <signals/logging.hpp>
#include <parameters/Persistence/TextSerializer.hpp>
#include <parameters/IVariableManager.h>
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

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
	LOG(error) << "Caught signal " << s;
    quit = true;
    if(s == 2)
        exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
	signal(SIGINT, sig_handler);
	signal(SIGILL, sig_handler);
	signal(SIGTERM, sig_handler);
    signal(SIGSEGV, sig_handler);
    
    boost::program_options::options_description desc("Allowed options");
	
    //boost::log::add_file_log(boost::log::keywords::file_name = "SimpleConsole%N.log", boost::log::keywords::rotation_size = 10 * 1024 * 1024);
    EagleLib::SetupLogging();
    
    Signals::thread_registry::get_instance()->register_thread(Signals::GUI);
    desc.add_options()
        ("file", boost::program_options::value<std::string>(), "Required - File to load for processing")
        ("config", boost::program_options::value<std::string>(), "Required - File containing node structure")
        ("plugins", boost::program_options::value<boost::filesystem::path>(), "Path to additional plugins to load")
		("log", boost::program_options::value<std::string>()->default_value("info"), "Logging verbosity. trace, debug, info, warning, error, fatal")
        ("mode", boost::program_options::value<std::string>()->default_value("interactive"), "Processing mode, options are interactive or batch")
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    
    
    {
        boost::posix_time::ptime initialization_start = boost::posix_time::microsec_clock::universal_time();
        LOG(info) << "Initializing GPU...";
        cv::cuda::GpuMat(10,10, CV_32F);
        boost::posix_time::ptime initialization_end= boost::posix_time::microsec_clock::universal_time();
        if(boost::posix_time::time_duration(initialization_end - initialization_start).total_seconds() > 1)
        {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, cv::cuda::getDevice());
            LOG(warning) << "Initialization took " << boost::posix_time::time_duration(initialization_end - initialization_start).total_milliseconds() << " ms.  CUDA code likely not generated for this architecture (" << props.major << "." << props.minor << ")";

        }
    }


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
#ifdef _MSC_VER
#ifdef _DEBUG
    currentDir = boost::filesystem::path(currentDir.string() + "/../Debug");
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/../RelWithDebInfo");
#endif
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/Plugins");
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
            }
        }
    }
    boost::thread gui_thread([]
    {
        Signals::thread_registry::get_instance()->register_thread(Signals::GUI);
        while(!boost::this_thread::interruption_requested())
        {
            Signals::thread_specific_queue::run();
            cv::waitKey(1);
        }
    });

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
    
        auto stream = EagleLib::IDataStream::create(document);
    
        auto nodes = EagleLib::NodeManager::getInstance().loadNodes(configFile);
        stream->AddNodes(nodes);

        LOG(info) << "Loaded " << nodes.size() << " top level nodes";
        for(int i = 0; i < nodes.size(); ++i)
        {
            PrintNodeTree(nodes[i].get(), 1);
        }
        stream->process();
    }else
    {
        std::vector<rcc::shared_ptr<EagleLib::IDataStream>> _dataStreams;
        EagleLib::IDataStream* current_stream = nullptr;
        EagleLib::Nodes::Node* current_node = nullptr;
		Parameters::Parameter* current_param = nullptr;

        auto print_options = []()->void
        {
            std::cout << 
                "- Options: \n"
				" - list_devices     -- List available connected devices for streaming access\n"
				" - load_file        -- Create a frame grabber for a document\n"
				"    document        -- Name of document to load\n"
                " - add              -- Add a node to the current selected object\n"
				"    node            -- name of node, names are listed with list\n"
                " - list             -- List constructable objects\n"
				"    nodes           -- List nodes registered\n"
				" - plugins          -- list all plugins\n"
                " - print            -- Prints the current streams, nodes in current stream, \n"
				"    streams         -- prints all streams\n"
				"    nodes           -- prints all nodes of current stream\n"
				"    parameters      -- prints all parameters of current node or stream\n"
				"    current         -- prints what is currently selected (default)\n"  
				"    signals         -- prints current signal map\n"
				"    inputs          -- prints possible inputs"
				" - set              -- Set a parameters value\n"
				"    name value      -- name value pair to be applied to parameter\n"
				" - select           -- Select object\n"
				"    name            -- select node by name\n"
				"    index           -- select stream by index\n"
				"    parameter       -- if currently a node is selected, select a parameter"
				" - delete           -- delete selected item\n"
				" - emit             -- Send a signal\n"
				"    name parameters -- signal name and parameters\n"
                " - save             -- Save node configuration\n"
                " - load             -- Load node configuration\n"
                " - help             -- Print this help\n"
                " - quit             -- Close program and cleanup\n"
                " - log              -- change logging level\n"
			    " - recompile        -- checks if any files need to be recompiled\n";
        };

        std::map<std::string, std::function<void(std::string)>> function_map;
		function_map["list_devices"] = [](std::string null)->void
		{
			auto constructors = EagleLib::ObjectManager::Instance().GetConstructorsForInterface(IID_FrameGrabber);
			for(auto constructor : constructors)
			{
				auto info = constructor->GetObjectInfo();
				if(info)
				{
					auto fg_info = dynamic_cast<EagleLib::FrameGrabberInfo*>(info);
					if(fg_info)
					{
						auto devices = fg_info->ListLoadableDocuments();
						if(devices.size())
						{
							std::stringstream ss;
							ss << fg_info->GetObjectName() << " can load:\n";
							for(auto& device : devices)
							{
								ss << "    " << device;
							}
							LOG(info) << ss.str();
						}
					}
				}
			}
		};
        function_map["load_file"] = [&_dataStreams](std::string doc)->void
        {
            if(EagleLib::IDataStream::CanLoadDocument(doc))
            {
                LOG(debug) << "Found a frame grabber which can load " << doc;
                auto stream = EagleLib::IDataStream::create(doc);
                if(stream->LoadDocument(doc))
                {
					stream->StartThread();
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
        function_map["quit"] = [](std::string)->void
        {
            quit = true;
        };
        function_map["print"] = [&_dataStreams, &current_stream, &current_node, &current_param](std::string what)->void
        {
            if(what == "streams")
            {
                for(auto& itr : _dataStreams)
                {
                    std::cout << " - " << itr->GetPerTypeId() << " - " << itr->GetFrameGrabber()->GetSourceFilename() << "\n";
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
                std::vector<Parameters::Parameter*> parameters;
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
                        Parameters::Persistence::Text::Serialize(&ss, itr);   
                        std::cout << " - " << itr->GetTreeName() << ": " << ss.str() << "\n";
                    }catch(...)
                    {
                        std::cout << " - " << itr->GetTreeName() << "\n";
                    }
                }
            }
            if(what == "current" || what.empty())
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
			if(what == "signals")
			{
				if (current_node)
				{
					current_node->GetDataStream()->GetSignalManager()->print_signal_map();
				}
				if (current_stream)
				{
					current_stream->GetSignalManager()->print_signal_map();
				}
			}
			if(what == "inputs")
			{
				if(current_param && current_node)
				{
					auto potential_inputs = current_node->GetVariableManager()->GetOutputParameters(current_param->GetTypeInfo());
					std::stringstream ss;
					if(potential_inputs.size())
					{
						ss << "Potential inputs: \n";
						for(auto& input : potential_inputs)
						{
							ss << " - " << input->GetName() << "\n";
						}
						LOG(info) << ss.str();
						return;
					}
					LOG(info) << "Unable to find any matching inputs for variable with name: " << current_param->GetName() << " with type: " << current_param->GetTypeInfo().name();
				}
				if(current_node)
				{
					auto params = current_node->getParameters();
					std::stringstream ss;
					for(auto param : params)
					{
						if(param->type & Parameters::Parameter::Input)
						{
							ss << " -- " << param->GetName() << " [ " << param->GetTypeInfo().name() << " ]\n";
							auto potential_inputs = current_node->GetVariableManager()->GetOutputParameters(param->GetTypeInfo());
							for(auto& input : potential_inputs)
							{
								ss << " - " << input->GetTreeName();
							}
						}
					}
					LOG(info) << ss.str();
				}
			}
		};
        function_map["select"] = [&_dataStreams,&current_stream, &current_node, &current_param](std::string what)
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
                    if(itr->GetPerTypeId() == idx)
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
			if(current_node)
			{
				auto child = current_node->getChild(what);
				if(child)
				{
					current_node = child.get();
					current_stream = nullptr;
					current_param = nullptr;
				}else
				{
					auto params = current_node->getParameters();
					for(auto& param : params)
					{
						if(param->GetName().find(what) != std::string::npos)
						{
							current_param = param;
							current_stream = nullptr;
						}
					}
				}
			}
        };
        function_map["delete"] = [&_dataStreams,&current_stream, &current_node](std::string what)
		{
			if(current_stream)
			{
				auto itr = std::find(_dataStreams.begin(), _dataStreams.end(), current_stream);
				if(itr != _dataStreams.end())
				{
					_dataStreams.erase(itr);
					current_stream = nullptr;
					LOG(info) << "Sucessfully deleted stream";
					return;
				}
			}else if(current_node)
			{
				if(auto parent = current_node->getParent())
				{
					parent->removeChild(current_node);
					current_node = nullptr;
					LOG(info) << "Sucessfully removed node from parent node";
					return;
				}else if(auto stream = current_node->GetDataStream())
				{
					stream->RemoveNode(current_node);
					current_node = nullptr;
					LOG(info) << "Sucessfully removed node from datastream";
					return;
				}
			}
			LOG(info) << "Unable to delete item";
		};
		function_map["help"] = [&print_options](std::string)->void{print_options();};
        function_map["list"] = [](std::string filter)->void
        {
            auto nodes = EagleLib::NodeManager::getInstance().getConstructableNodes();
            for(auto& node : nodes)
            {
                if(filter.size())
                {
                    if(node.find(filter) != std::string::npos)
                    {
                        std::cout << " - " << node << "\n";
                    }
                }else
                {
                    std::cout << " - " << node << "\n";
                }
                
            }
        };
		function_map["plugins"] = [](std::string null)->void
		{
			auto plugins = EagleLib::ListLoadedPlugins();
			std::stringstream ss;
			ss << "Loaded / failed plugins:\n";
			for(auto& plugin: plugins)
			{
				ss << "  " << plugin << "\n";
			}
			LOG(info) << ss.str();
		};
        function_map["add"] = [&current_node, &current_stream](std::string name)->void
        {
			if(current_stream)
			{
				current_stream->AddNode(name);
				return;
			}
			if(current_node)
			{
				EagleLib::NodeManager::getInstance().addNode(name, current_node);
			}
        };
		function_map["set"] = [&current_node, &current_stream, &current_param](std::string value)->void
		{
			if(current_param && current_node && current_param->type & Parameters::Parameter::Input)
			{
				auto variable_manager = current_node->GetVariableManager();
				auto output = variable_manager->GetOutputParameter(value);
				if(output)
				{
					variable_manager->LinkParameters(output, current_param);
					return;
				}
			}
			if(current_param)
			{
				Parameters::Persistence::Text::DeSerialize(&value, current_param);
			}
			if (current_node)
			{
				auto params = current_node->getParameters();
				for(auto& param : params)
				{
					auto pos = value.find(param->GetName());
					if(pos != std::string::npos)
					{
						LOG(info) << "Setting value for parameter " << param->GetName() << " to " << value.substr(pos);
						std::stringstream ss;
						ss << value.substr(pos);
						Parameters::Persistence::Text::DeSerialize(&ss, param);
						return;
					}
				}
				LOG(info) << "Unable to find parameter by name for set string: " << value;
			}else if(current_stream)
			{
				auto params = current_stream->GetFrameGrabber()->getParameters();
				for(auto& param : params)
				{
					auto pos = value.find(param->GetName());
					if(pos != std::string::npos)
					{
						auto len = param->GetName().size() + 1;
						LOG(info) << "Setting value for parameter " << param->GetName() << " to " << value.substr(len);
						std::stringstream ss;
						ss << value.substr(len);
						Parameters::Persistence::Text::DeSerialize(&ss, param);
						return;
					}
				}
				LOG(info) << "Unable to find parameter by name for set string: " << value;
			}
		};
		function_map["ls"] = [&function_map](std::string str)->void {function_map["print"](str); };
		function_map["emit"] = [&current_node, &current_stream](std::string name)
		{
			EagleLib::SignalManager* mgr = nullptr;
			if (current_node)
			{
				mgr = current_node->GetDataStream()->GetSignalManager();
			}
			if (current_stream)
			{
				mgr = current_stream->GetSignalManager();
			}
			std::vector<Signals::signal_base*> signals;
			if (mgr)
			{
				signals = mgr->get_signals(name);
			}
			auto table = PerModuleInterface::GetInstance()->GetSystemTable();
			if (table)
			{
				auto global_signal_manager = table->GetSingleton<EagleLib::SignalManager>();
				if(global_signal_manager)
				{
					auto global_signals = global_signal_manager->get_signals(name);
					signals.insert(signals.end(), global_signals.begin(), global_signals.end());
				}
			}
			if (signals.size() == 0)
			{
				LOG(info) << "No signals found with name: " << name;
				return;
			}
			int idx = 0;
			if (signals.size() > 1)
			{
				for (auto signal : signals)
				{
					std::cout << idx << " - " << signal->get_signal_type().name();
					++idx;
				}
				std::cin >> idx;
			}
			LOG(info) << "Attempting to send signal with name \"" << name << "\" and signature: " << signals[idx]->get_signal_type().name();
            auto proxy = Signals::serialization::text::factory::instance()->get_proxy(signals[idx]);
            if(proxy)
            {
                proxy->send(signals[idx], "");
				delete proxy;
            }
			
			
		};
        function_map["log"] = [](std::string level)
        {
        if (level == "trace")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::trace);
		if (level == "debug")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
		if (level == "info")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
		if (level == "warning")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
		if (level == "error")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::error);
		if (level == "fatal")
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::fatal);
        };
		bool swap_required = false;
		function_map["recompile"] = [&swap_required, &_dataStreams](std::string null)
		{
			if(swap_required)
			{
				if(EagleLib::ObjectManager::Instance().CheckRecompile(true))
				{
					LOG(info) << "Still compiling";
				}else
				{
					LOG(info) << "Recompile complete";
					swap_required = false;
					for(auto& stream : _dataStreams)
					{
						stream->StartThread();
					}
				}
			}else
			{
			
			}
			if(EagleLib::ObjectManager::Instance().CheckRecompile(false))
			{
				swap_required = true;
				for(auto& stream : _dataStreams)
				{
					stream->StopThread();
				}
				LOG(info) << "Recompiling.....";
			}
		};
		if (vm.count("file"))
		{
			function_map["load_file"](vm["file"].as<std::string>());
		}
		
		print_options();
        
		while(!quit)
        {
            std::string command_line;
            std::getline(std::cin, command_line);
	        std::stringstream ss;
            ss << command_line;
            std::string command;
            std::getline(ss, command, ' ');
            if(function_map.count(command))
            {
                std::string rest;
				std::getline(ss, rest);
                LOG(debug) << "Executing command (" << command << ") with arguments: " << rest;
                function_map[command](rest);
            }else
            {
                LOG(warning) << "Invalid command: " << command_line;
                print_options();
            }
            for(int i = 0; i < 20; ++i)
                Signals::thread_specific_queue::run_once();
        }
        LOG(info) << "Shutting down";
    }
    gui_thread.interrupt();
    gui_thread.join();
    
    return 0;
}
