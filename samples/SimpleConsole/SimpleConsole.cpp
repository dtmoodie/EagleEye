
#include <Aquila/core/Aquila.hpp>
#include <Aquila/core/Logging.hpp>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/core/IDataStream.hpp>
#include <Aquila/nodes/NodeFactory.hpp>
#include <Aquila/gui/UiCallbackHandlers.h>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>

#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/RelayManager.hpp>
#include <MetaObject/params/IVariableManager.hpp>
#include <MetaObject/serialization/SerializationFactory.hpp>
#include <MetaObject/logging/Profiling.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/thread/ThreadPool.hpp>
#include <MetaObject/params/buffers/IBuffer.hpp>
#include <MetaObject/logging/Profiling.hpp>
#include <RuntimeObjectSystem/RuntimeObjectSystem.h>

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
#include <boost/date_time.hpp>
#include <signal.h> // SIGINT, etc

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "MetaObject/MetaParameters.hpp"
#include "Aquila/rcc/SystemTable.hpp"

#ifdef HAVE_WT
#include "vclick.hpp"
#endif

#include <fstream>

void PrintNodeTree(aq::Nodes::Node* node, int depth)
{
    for(int i = 0; i < depth; ++i)
    {
        std::cout << "=";
    }
    std::cout << node->getTreeName() << std::endl;
    auto children = node->getChildren();
    for(int i = 0; i < children.size(); ++i)
    {
        PrintNodeTree(children[i].get(), depth + 1);
    }
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& strs)
{
    for( const auto & str : strs)
    {
        os << str << "\n ";
    }
    return os;
}

void PrintBuffers(aq::Nodes::Node* node, std::vector<std::string>& printed_nodes)
{
    std::string name = node->getTreeName();
    if(std::find(printed_nodes.begin(), printed_nodes.end(), name) != printed_nodes.end())
    {
        return;
    }
    printed_nodes.push_back(name);
    std::vector<mo::InputParam*> inputs = node->getInputs();
    std::cout << "--------\n" << name << std::endl;
    for(mo::InputParam* input : inputs)
    {
        mo::IParam* param = input->getInputParam();
        mo::Buffer::IBuffer* buf = dynamic_cast<mo::Buffer::IBuffer*>(param);
        if(buf)
        {
            std::cout << param->getTreeName() << " - " << buf->getSize() << std::endl;
        }
    }

    auto children = node->getChildren();
    for(auto child : children)
    {
        PrintBuffers(child.get(), printed_nodes);
    }
}

static volatile bool quit;

void sig_handler(int s)
{
    switch(s)
    {
    case SIGSEGV:
    {
        //std::cout << "Caught SIGSEGV " << mo::print_callstack(2, true);
        break;
    }
    case SIGINT:
    {
        //std::cout << "Caught SIGINT " << mo::print_callstack(2, true);
        std::cout << "Caught SIGINT, shutting down" << std::endl;
        quit = true;
        return;
    }
    case SIGILL:
    {
        //std::cout  << "Caught SIGILL " << mo::print_callstack(2, true);
        break;
    }
    case SIGTERM:
    {
        //std::cout  << "Caught SIGTERM " << mo::print_callstack(2, true);
        break;
    }
#ifndef _MSC_VER
    case SIGKILL:
    {
        //std::cout  << "Caught SIGKILL " << mo::print_callstack(2, true);
        break;
    }
#endif
    }
}

int main(int argc, char* argv[])
{
    boost::program_options::options_description desc("Allowed options");
    SystemTable table;
    mo::MetaObjectFactory::instance(&table);

    desc.add_options()
        ("file", boost::program_options::value<std::string>(), "Optional - File to load for processing")
        ("config", boost::program_options::value<std::string>(), "Optional - File containing node structure")
        ("launch", boost::program_options::value<std::string>(), "Optional - File containing node structure")
        ("plugins", boost::program_options::value<boost::filesystem::path>(), "Path to additional plugins to load")
        ("log", boost::program_options::value<std::string>()->default_value("info"), "Logging verbosity. trace, debug, info, warning, error, fatal")
        ("log-dir", boost::program_options::value<std::string>(), "directory for log output")
        ("mode", boost::program_options::value<std::string>()->default_value("interactive"), "Processing mode, options are interactive or batch")
        ("script,s", boost::program_options::value<std::string>(), "Text file with scripting commands")
        ("profile,p", boost::program_options::bool_switch(), "Profile application")
        ("gpu", boost::program_options::value<int>()->default_value(0), "")
        ("docroot", boost::program_options::value<std::string>(), "")
        ("http-address", boost::program_options::value<std::string>(), "")
        ("http-port", boost::program_options::value<std::string>(), "")
        ("disable-rcc", boost::program_options::bool_switch(), "Disable rcc")
        ("quit-on-eos", boost::program_options::bool_switch(), "Quit program on end of stream signal")
        ("disable-input", boost::program_options::bool_switch(), "Disable input for batch scripting, and nvprof")
        ("profile-for", boost::program_options::value<int>(), "Amount of time to run before quitting, use with profiler")
        ;

    boost::program_options::variables_map vm;
    auto parsed_options = boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    boost::program_options::store(parsed_options, vm);
    mo::MetaParams::initialize();

    if(vm.count("log-dir"))
        aq::Init(vm["log-dir"].as<std::string>());
    else
        aq::Init();

    mo::MetaObjectFactory::instance()->registerTranslationUnit();

    auto g_allocator = mo::Allocator::getThreadSafeAllocator();
    cv::cuda::GpuMat::setDefaultAllocator(g_allocator);
    cv::Mat::setDefaultAllocator(g_allocator);

    g_allocator->setName("Global Allocator");
    if(!mo::GpuThreadAllocatorSetter<cv::cuda::GpuMat>::Set(g_allocator))
    {
        LOG(info) << "Unable to set thread specific gpu allocator in opencv";
    }
    if(!mo::CpuThreadAllocatorSetter<cv::Mat>::Set(g_allocator))
    {
        LOG(info) << "Unable to set thread specific cpu allocator in opencv";
    }

    auto unrecognized = boost::program_options::collect_unrecognized(parsed_options.options, boost::program_options::include_positional);
    std::map<std::string, std::string> replace_map;
    std::map<std::string, std::string> variable_replace_map;
    for(auto& option : unrecognized)
    {
        auto pos = option.find(":=");
        if(pos != std::string::npos)
        {
            std::string start = option.substr(0, pos);
            std::string end = option.substr(pos+2);
            if(end[0] == '\"' && end[end.size() - 1] == '\"')
            {
                end = end.substr(1, end.size() - 2);
            }
            replace_map["${" + start + "}"] = end;
            continue;
        }
        pos = option.find("=");
        if(option.find("--") == 0 && pos != std::string::npos)
        {
            std::string var_name = option.substr(2, pos - 2);
            std::string var_value = option.substr(pos + 1);
            if (var_value[0] == '\"' && var_value[var_value.size() - 1] == '\"')
            {
                var_value = var_value.substr(1, var_value.size() - 2);
            }
            variable_replace_map[var_name] = var_value;
        }
    }
    if(replace_map.size())
    {
        std::stringstream ss;
        for(const auto& pair : replace_map)
            ss << "\n" << pair.first << " = " << pair.second;
        LOG(debug) << "Input string replacements: " << ss.str();
    }

    if(variable_replace_map.size())
    {
        std::stringstream ss;
        for(const auto& pair : variable_replace_map)
            ss << "\n" <<  pair.first << " = " << pair.second;
        LOG(debug) << "Input variable replacements: " << ss.str();
    }

    if(vm["profile"].as<bool>())
    {
        mo::initProfiling();
    }
    cv::cuda::setDevice(vm["gpu"].as<int>());
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
    boost::filesystem::path currentDir = boost::filesystem::path(argv[0]).parent_path();
#ifdef _MSC_VER
    currentDir = boost::filesystem::path(currentDir.string());
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/Plugins");
#endif
    LOG(info) << "Looking for plugins in: " << currentDir.string();
    boost::filesystem::directory_iterator end_itr;
    if(boost::filesystem::is_directory(currentDir))
    {
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
                    mo::MetaObjectFactory::instance()->loadPlugin(file);
                }
            }
        }
    }
    boost::thread gui_thread([]
    {
        mo::ThreadRegistry::instance()->registerThread(mo::ThreadRegistry::GUI);
        while(!boost::this_thread::interruption_requested())
        {
            mo::setThreadName("SimpleConsole GUI thread");
            try
            {
                mo::ThreadSpecificQueue::run();
            }
            catch (boost::thread_interrupted& err)
            {
                break;
            }
            catch(...)
            {
                LOG(debug) << "Unknown / unhandled exception thrown in gui thread event handler";
            }
            try
            {
                //cv::waitKey(1);
                aq::WindowCallbackHandler::EventLoop::Instance()->run();
            }catch(mo::ExceptionWithCallStack<cv::Exception>& e)
            {

            }catch(cv::Exception&e)
            {

            }
            catch(boost::thread_interrupted& err)
            {
                break;
            }catch(...)
            {

            }
        }
        LOG(info) << "Gui thread shutting down naturally";
    });
    mo::RelayManager manager;

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
                    mo::MetaObjectFactory::instance()->loadPlugin(file);
                }
            }
        }
    }

    if(vm["mode"].as<std::string>() == "batch")
    {

    }else
    {
        std::vector<rcc::shared_ptr<aq::IDataStream>> _dataStreams;
        rcc::weak_ptr<aq::IDataStream> current_stream;
        rcc::weak_ptr<aq::Nodes::Node> current_node;
        mo::IParam* current_param = nullptr;

        auto print_options = []()->void
        {
            std::cout <<
                "- Options: \n"
                " - list_devices     -- List available connected devices for streaming access\n"
                " - load_file        -- create a frame grabber for a document\n"
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
                "    inputs          -- prints possible inputs\n"
                " - info             -- get info on given node\n"
                " - set              -- Set a parameters value\n"
                "    name value      -- name value pair to be applied to parameter\n"
                " - select           -- Select object\n"
                "    name            -- select node by name\n"
                "    index           -- select stream by index\n"
                "    parameter       -- if currently a node is selected, select a parameter\n"
                " - delete           -- delete selected item\n"
                " - emit             -- Send a signal\n"
                "    name parameters -- signal name and parameters\n"
                " - save             -- Save node configuration\n"
                " - load             -- Load node configuration\n"
                " - help             -- Print this help\n"
                " - quit             -- Close program and cleanup\n"
                " - log              -- change logging level\n"
                " - link             -- add link directory\n"
                " - recompile        \n"
                "   check            -- checks if any files need to be recompiled\n"
                "   swap             -- swaps any objects that were recompiled\n";
        };
        std::vector<std::shared_ptr<mo::Connection>> connections;
        std::vector<std::shared_ptr<mo::ISlot>> _slots;
        std::vector<std::pair<std::string, std::string>> documents_list;
        mo::TSlot<void(std::string)>* slot;
        slot = new mo::TSlot<void(std::string)>(
            std::bind([&documents_list](std::string null)->void
            {
                documents_list.clear();
                auto constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::Nodes::IFrameGrabber::s_interfaceID);
                int index = 0;
                for(auto constructor : constructors)
                {
                    auto fg_info = dynamic_cast<aq::Nodes::IFrameGrabber::InterfaceInfo*>(constructor->GetObjectInfo());
                    if(fg_info)
                    {
                        auto documents = fg_info->listLoadablePaths();
                        for(auto& document : documents)
                        {
                            std::cout << " - " << index << "  [" << fg_info->getObjectName() << "] " << document << "\n";
                            documents_list.emplace_back(document, fg_info->getDisplayName());
                            ++index;
                        }
                    }
                }
            }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "list_devices"));


        slot = new mo::TSlot<void(std::string)>(
            std::bind([&_dataStreams, &documents_list](std::string doc)->void
        {
            std::string fg_override;
            int index = -1;
            if(!boost::conversion::detail::try_lexical_convert(doc, index))
            {
                index = -1;
            }
            if(index != -1 && index >= 0 && index < documents_list.size())
            {
                doc = documents_list[index].first;
                fg_override = documents_list[index].second;
            }
            auto ds = aq::IDataStream::create(doc, fg_override);
            if(ds)
            {
                ds->startThread();
                _dataStreams.push_back(ds);
            }
        }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "load_file"));

        slot = new mo::TSlot<void(std::string)>(
            std::bind([](std::string)->void
        {
            quit = true;
        }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "quit"));


        auto func = [&_dataStreams, &current_stream, &current_node, &current_param](std::string what)->void
        {
            if(what == "streams")
            {
                for(auto& itr : _dataStreams)
                {
                    auto fgs = itr->getTopLevelNodes();
                    for(auto& fg : fgs)
                    {
                        if(auto frame_grabber = fg.DynamicCast<aq::Nodes::IFrameGrabber>())
                        {
                            std::cout << " - " << frame_grabber->GetPerTypeId() << " - " << frame_grabber->loaded_document << "\n";
                        }
                        else
                        {

                        }
                    }

                }
                if(_dataStreams.empty())
                    std::cout << "No streams exist\n";
            }
            if(what == "nodes")
            {
                if(current_stream)
                {
                    auto nodes = current_stream->getNodes();
                    for(auto& node : nodes)
                    {
                        PrintNodeTree(node.get(), 0);
                    }
                }
                else if(current_node)
                {
                    PrintNodeTree(current_node.get(), 0);
                }
            }
            if(what == "parameters")
            {
                std::vector<mo::IParam*> parameters;
                if(current_node)
                {
                    parameters = current_node->getAllParams();
                }
                if(current_stream)
                {

                }
                for(auto& itr : parameters)
                {
                    std::stringstream ss;
                    try
                    {
                        if(itr->checkFlags(mo::Input_e))
                        {
                            if(auto input = dynamic_cast<mo::InputParam*>(itr))
                            {
                                std::stringstream ss;
                                ss << " - " << itr->getTreeName() << " [";
                                auto input_param = input->getInputParam();
                                if(input_param)
                                {
                                    ss << input_param->getTreeName();
                                }else
                                {
                                    ss << "input not set";
                                }
                                ss << "]\n";
                                std::cout << ss.str();
                            }
                        }
                        else
                        {
                            auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(itr->getTypeInfo());
                            if (func)
                            {
                                std::stringstream ss;
                                ss << " - " << itr->getTreeName() << " [";
                                func(itr, ss);
                                std::cout << ss.str() << "]\n";
                            }
                            else
                            {
                                std::cout << " - " << itr->getTreeName() << "\n";
                            }
                        }
                    }catch(...)
                    {
                        //std::cout << " - " << itr->getTreeName() << "\n";
                    }
                }
                if(parameters.empty())
                    std::cout << "No parameters exist\n";
            }
            if(what == "current" || what.empty())
            {
                if(current_stream)
                {
                    auto fgs = current_stream->getTopLevelNodes();
                    for(auto& fg : fgs)
                    {
                        if(auto f_g = fg.DynamicCast<aq::Nodes::IFrameGrabber>())
                        {
                            std::cout << " - Datasource: " << f_g->loaded_document << "\n";
                        }
                    }
                }
                if(current_node)
                {
                    std::cout << " - Current node: " << current_node->getTreeName() << "\n";
                }
                if(current_param)
                {
                    std::cout << " - Current parameter: " << current_param->getTreeName() << "\n";
                }
                if(!current_node && !current_param && !current_stream)
                    std::cout << "Nothing currently selected\n";
            }
            if(what == "signals")
            {
                if (current_node)
                {
                    std::vector<mo::SignalInfo*> infos;
                    current_node->getSignalInfo(infos);
                    for(auto& info : infos)
                    {
                        std::cout << info->print();
                    }
                }
                if (current_stream)
                {
                    //current_stream->GetSignalManager()->print_signal_map();
                }
            }
            if(what == "inputs")
            {
                if(current_param && current_node)
                {
                    auto potential_inputs = current_node->getDataStream()->getVariableManager()->getOutputParams(current_param->getTypeInfo());
                    std::stringstream ss;
                    if(potential_inputs.size())
                    {
                        ss << "Potential inputs: \n";
                        for(auto& input : potential_inputs)
                        {
                            ss << " - " << input->getTreeName() << "\n";
                        }
                        std::cout << ss.str() << std::endl;
                        return;
                    }
                    std::cout << "Unable to find any matching inputs for variable with name: " << current_param->getName() << " with type: " << current_param->getTypeInfo().name() << std::endl;
                }
                if(current_node)
                {
                    auto params = current_node->getAllParams();
                    std::stringstream ss;
                    for(auto param : params)
                    {
                        if(param->checkFlags(mo::Input_e))
                        {
                            ss << " -- " << param->getTreeName() << " [ " << param->getTypeInfo().name() << " ]\n";
                            auto potential_inputs = current_node->getDataStream()->getVariableManager()->getOutputParams(param->getTypeInfo());
                            for(auto& input : potential_inputs)
                            {
                                ss << " - " << input->getTreeName();
                            }
                        }
                    }
                    std::cout << ss.str() << std::endl;
                }
            }
            if(what == "projects")
            {
                THROW(debug) << "Needs to be reimplemented";
                //auto project_count = aq::ObjectManager::Instance().getProjectCount();
                /*std::stringstream ss;
                ss << "\n";
                for(int i = 0; i < project_count; ++i)
                {
                    ss << i << " - " << aq::ObjectManager::Instance().getProjectName(i) << "\n";
                }
                std::cout << ss.str() << std::endl;*/
            }
            if(what == "plugins")
            {
                auto plugins = mo::MetaObjectFactory::instance()->listLoadedPlugins();
                std::stringstream ss;
                ss << "\n";
                for(auto& plugin : plugins)
                {
                    ss << plugin << "\n";
                }
                if(plugins.empty())
                    ss << "No plugins loaded\n";
                std::cout << ss.str() << std::endl;
            }
            if(what == "tree")
            {
                if (current_stream)
                {
                    auto nodes = current_stream->getNodes();
                    for (auto& node : nodes)
                    {
                        PrintNodeTree(node.get(), 0);
                    }
                }
                else if (current_node)
                {
                    auto nodes = current_node->getDataStream()->getNodes();
                    for(auto node : nodes)
                        PrintNodeTree(node.get(), 0);
                }
            }
            if(what == "buffers")
            {
                if(current_stream)
                {
                    auto nodes = current_stream->getNodes();
                    std::vector<std::string> printed;
                    for(auto node : nodes)
                    {
                        PrintBuffers(node.get(), printed);
                    }
                }
            }
        };
        slot = new mo::TSlot<void(std::string)>(std::bind(func, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "print"));
        connections.push_back(manager.connect(slot, "ls"));
        slot = new mo::TSlot<void(std::string)>(
                    std::bind([](std::string obj)
        {
            IObjectConstructor* constructor = mo::MetaObjectFactory::instance()->getConstructor(obj.c_str());
            if(constructor)
            {
                mo::IMetaObjectInfo* info = dynamic_cast<mo::IMetaObjectInfo*>(constructor->GetObjectInfo());
                if(info)
                {
                    std::cout << info->print();
                }
            }else
            {
                std::cout << "No constructor found for " << obj;
            }
        }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "info"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
        [&current_stream, &current_node, &variable_replace_map, &replace_map](std::string file)
        {
            if (current_stream)
            {
                //current_stream->SaveStream(file);
                rcc::shared_ptr<aq::IDataStream> stream(current_stream);
                std::vector<rcc::shared_ptr<aq::IDataStream>> streams;
                streams.push_back(stream);
                aq::IDataStream::save(file, streams, variable_replace_map, replace_map);
                stream->startThread();
            }
            else if (current_node)
            {

            }
        }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "save"));

        slot = new mo::TSlot<void(std::string)>(std::bind([](std::string null)
        {
            //mo::InitProfiling();
        }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "profile"));

        bool quit_on_eos = vm["quit-on-eos"].as<bool>();
        mo::TSlot<void()> eos_slot(std::bind([]()
        {
            LOG_FIRST_N(info, 1) << "End Of Stream received, shutting down";
            quit = true;
        }));
        std::vector<std::shared_ptr<mo::Connection>> eos_connections;
        slot = new mo::TSlot<void(std::string)>(
        std::bind([&_dataStreams, &current_stream, &current_node, quit_on_eos, &eos_connections, &eos_slot, &variable_replace_map, &replace_map](std::string file)
        {
            auto streams = aq::IDataStream::load(file, variable_replace_map, replace_map);

            if(streams.size())
            {
                for(auto& stream : streams)
                {
                    stream->startThread();
                    _dataStreams.push_back(stream);
                    if(quit_on_eos)
                    {
                        stream->getRelayManager()->connect(&eos_slot, "eos");
                    }
                }
            }else
            {
                std::cout << "Load of " << file << " failed";
            }
        }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "load"));

        slot = new mo::TSlot<void(std::string)>(
            std::bind([&_dataStreams,&current_stream, &current_node, &current_param](std::string what)
        {
            if(what == "null")
            {
                current_stream.reset();
                current_node.reset();
                current_param = nullptr;
            }
            int idx = -1;
            std::string name;
            if(!boost::conversion::detail::try_lexical_convert(what, idx))
            {
                idx = -1;
            }

            if(idx == -1)
            {
                name = what;
            }
            if(idx != -1)
            {
                std::cout << "Selecting stream " << idx << std::endl;
                for(auto& itr : _dataStreams)
                {
                    if(itr != nullptr)
                    {
                        if(itr->GetPerTypeId() == idx)
                        {
                            current_stream = itr.get();
                            current_node.reset();
                            current_param = nullptr;
                            auto nodes = current_stream->getNodes();
                            for (auto& node : nodes)
                            {
                                PrintNodeTree(node.get(), 0);
                            }
                            return;
                        }
                    }
                }
                std::cout << "No stream found by given index " << idx << "\n";
            }
            if(current_stream)
            {
                // look for a node with this name, relative then look absolute
                auto node = current_stream->getNode(what);
                if(node)
                {
                    current_node = node;
                }
                std::cout << "No node found with given name " << what << std::endl;
            }
            if(current_node)
            {
                auto child = current_node->getChild(what);
                if(child)
                {
                    current_node = child.get();
                    current_stream.reset();
                    current_param = nullptr;
                    std::cout << "Successfully set node to " << child->getTreeName() << "\n";
                    std::vector<mo::IParam*> parameters;
                    if(current_node)
                    {
                        parameters = current_node->getAllParams();
                    }
                    if(current_stream)
                    {

                    }
                    for(auto& itr : parameters)
                    {
                        std::stringstream ss;
                        try
                        {
                            if(itr->checkFlags(mo::Input_e))
                            {
                                if(auto input = dynamic_cast<mo::InputParam*>(itr))
                                {
                                    std::stringstream ss;
                                    ss << " - " << itr->getTreeName() << " [";
                                    auto input_param = input->getInputParam();
                                    if(input_param)
                                    {
                                        ss << input_param->getTreeName();
                                    }else
                                    {
                                        ss << "input not set";
                                    }
                                    ss << "]\n";
                                    std::cout << ss.str();
                                }
                            }
                            else
                            {
                                auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(itr->getTypeInfo());
                                if (func)
                                {
                                    std::stringstream ss;
                                    ss << " - " << itr->getTreeName() << " [";
                                    func(itr, ss);
                                    std::cout << ss.str() << "]\n";
                                }
                                else
                                {
                                    std::cout << " - " << itr->getTreeName() << "\n";
                                }
                            }
                        }catch(...)
                        {
                            //std::cout << " - " << itr->getTreeName() << "\n";
                        }
                    }
                    if(parameters.empty())
                        std::cout << "No parameters exist\n";
                    return;
                } else
                {
                    auto stream = current_node->getDataStream();
                    if(auto node = stream->getNode(what))
                    {
                        current_node = node;
                        current_stream.reset();
                        current_param = nullptr;
                        std::cout << "Successfully set node to " << node->getTreeName() << "\n";
                        std::vector<mo::IParam*> parameters;
                        if(current_node)
                        {
                            parameters = current_node->getAllParams();
                        }
                        if(current_stream)
                        {

                        }
                        for(auto& itr : parameters)
                        {
                            std::stringstream ss;
                            try
                            {
                                if(itr->checkFlags(mo::Input_e))
                                {
                                    if(auto input = dynamic_cast<mo::InputParam*>(itr))
                                    {
                                        std::stringstream ss;
                                        ss << " - " << itr->getTreeName() << " [";
                                        auto input_param = input->getInputParam();
                                        if(input_param)
                                        {
                                            ss << input_param->getTreeName();
                                        }else
                                        {
                                            ss << "input not set";
                                        }
                                        ss << "]\n";
                                        std::cout << ss.str();
                                    }
                                }
                                else
                                {
                                    auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(itr->getTypeInfo());
                                    if (func)
                                    {
                                        std::stringstream ss;
                                        ss << " - " << itr->getTreeName() << " [";
                                        func(itr, ss);
                                        std::cout << ss.str() << "]\n";
                                    }
                                    else
                                    {
                                        std::cout << " - " << itr->getTreeName() << "\n";
                                    }
                                }
                            }catch(...)
                            {
                                //std::cout << " - " << itr->getTreeName() << "\n";
                            }
                        }
                        if(parameters.empty())
                            std::cout << "No parameters exist\n";
                        return;
                    }
                    else
                    {
                        auto params = current_node->getAllParams();
                        for(auto& param : params)
                        {
                            std::string name = param->getName();
                            auto pos = name.find(':');
                            if(pos == std::string::npos)
                            {
                                if(name == what)
                                {
                                    current_param = param;
                                    current_stream.reset();
                                    return;
                                }
                            }else
                            {
                                if(name.substr(pos+1) == what)
                                {
                                    current_param = param;
                                    current_stream.reset();
                                    return;
                                }
                            }
                        }
                        for(auto& param : params)
                        {
                            if(param->getName().find(what) != std::string::npos)
                            {
                                current_param = param;
                                current_stream.reset();
                                return;
                            }
                        }
                        std::cout << "No parameter found with given name " << what << "\n";
                    }
                }
            }
        }, std::placeholders::_1));


        connections.push_back(manager.connect(slot,"select"));


        slot = new mo::TSlot<void(std::string)>(
            std::bind([&_dataStreams,&current_stream, &current_node](std::string what)
        {
            if(current_stream)
            {
                auto itr = std::find(_dataStreams.begin(), _dataStreams.end(), current_stream.get());
                if(itr != _dataStreams.end())
                {
                    _dataStreams.erase(itr);
                    current_stream.reset();
                    std::cout << "Sucessfully deleted stream\n";
                    return;
                }
            }else if(current_node)
            {
                auto parents = current_node->getParents();
                if(parents.size())
                {
                    for(auto parent : parents)
                    {
                        parent->removeChild(current_node.get());
                    }
                    current_node.reset();
                    std::cout << "Sucessfully removed node from parent node\n";
                    return;
                }else if (auto stream = current_node->getDataStream())
                {
                    stream->removeNode(current_node.get());
                    current_node.reset();
                    std::cout << "Sucessfully removed node from datastream\n";
                    return;
                }
            }
            std::cout << "Unable to delete item\n";
        }, std::placeholders::_1));

        connections.push_back(manager.connect(slot, "delete"));

        slot = new mo::TSlot<void(std::string)>(std::bind([&print_options](std::string)->void {print_options(); }, std::placeholders::_1));

        connections.push_back(manager.connect(slot,"help" ));

        slot = new mo::TSlot<void(std::string)>(std::bind([](std::string filter)->void
        {
            if (filter.size())
            {
                auto nodes = aq::NodeFactory::Instance()->GetConstructableNodes();
                for(auto& node : nodes)
                {
                    if(node.find(filter) != std::string::npos)
                    {
                        std::cout << " - " << node << "\n";
                    }
                }
            }else
            {
                auto constructors = mo::MetaObjectFactory::instance()->getConstructors();
                std::map<std::string, std::vector<IObjectConstructor*>> interface_map;
                for(auto constructor : constructors)
                {
                    IObjectInfo* info = constructor->GetObjectInfo();
                    if(info)
                    {
                        interface_map[info->getInterfaceName()].push_back(constructor);
                    }
                }
                for(auto itr = interface_map.begin(); itr != interface_map.end(); ++itr)
                {
                    std::cout << "========= " << itr->first << std::endl;
                    for(auto ctr : itr->second)
                    {
                        std::cout << "  " << ctr->GetObjectInfo()->getObjectName() << std::endl;
                    }
                }
            }
        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "list"));

        slot = new mo::TSlot<void(std::string)>(std::bind([](std::string null)->void
        {
            auto plugins = mo::MetaObjectFactory::instance()->listLoadedPlugins();
            std::stringstream ss;
            ss << "Loaded / failed plugins:\n";
            for(auto& plugin: plugins)
            {
                ss << "  " << plugin << "\n";
            }
            std::cout << ss.str() << std::endl;;
        }, std::placeholders::_1));


        connections.push_back(manager.connect(slot, "plugins"));

        slot = new mo::TSlot<void(std::string)>(std::bind([&current_node, &current_stream](std::string name)->void
        {
            if(current_stream)
            {
                auto added_nodes = current_stream->addNode(name);
                if(added_nodes.size())
                    current_node = added_nodes[0];
                std::vector<mo::IParam*> parameters;
                if(current_node)
                {
                    parameters = current_node->getAllParams();
                }
                for(auto& itr : parameters)
                {
                    std::stringstream ss;
                    try
                    {
                        if(itr->checkFlags(mo::Input_e))
                        {
                            if(auto input = dynamic_cast<mo::InputParam*>(itr))
                            {
                                std::stringstream ss;
                                ss << " - " << itr->getTreeName() << " [";
                                auto input_param = input->getInputParam();
                                if(input_param)
                                {
                                    ss << input_param->getTreeName();
                                }else
                                {
                                    ss << "input not set";
                                }
                                ss << "]\n";
                                std::cout << ss.str();
                            }
                        }
                        else
                        {
                            auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(itr->getTypeInfo());
                            if (func)
                            {
                                std::stringstream ss;
                                ss << " - " << itr->getTreeName() << " [";
                                func(itr, ss);
                                std::cout << ss.str() << "]\n";
                            }
                            else
                            {
                                std::cout << " - " << itr->getTreeName() << "\n";
                            }
                        }
                    }catch(...)
                    {
                        //std::cout << " - " << itr->getTreeName() << "\n";
                    }
                }
                if(parameters.empty())
                    std::cout << "No parameters exist\n";
                return;
            }
            if(current_node)
            {
                auto added_nodes = aq::NodeFactory::Instance()->addNode(name, current_node.get());
                if(added_nodes.size() == 1)
                {
                    current_node = added_nodes[0];
                }
                std::vector<mo::IParam*> parameters;
                if(current_node)
                {
                    parameters = current_node->getAllParams();
                }
                for(auto& itr : parameters)
                {
                    std::stringstream ss;
                    try
                    {
                        if(itr->checkFlags(mo::Input_e))
                        {
                            if(auto input = dynamic_cast<mo::InputParam*>(itr))
                            {
                                std::stringstream ss;
                                ss << " - " << itr->getTreeName() << " [";
                                auto input_param = input->getInputParam();
                                if(input_param)
                                {
                                    ss << input_param->getTreeName();
                                }else
                                {
                                    ss << "input not set";
                                }
                                ss << "]\n";
                                std::cout << ss.str();
                            }
                        }
                        else
                        {
                            auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(itr->getTypeInfo());
                            if (func)
                            {
                                std::stringstream ss;
                                ss << " - " << itr->getTreeName() << " [";
                                func(itr, ss);
                                std::cout << ss.str() << "]\n";
                            }
                            else
                            {
                                std::cout << " - " << itr->getTreeName() << "\n";
                            }
                        }
                    }catch(...)
                    {
                        //std::cout << " - " << itr->getTreeName() << "\n";
                    }
                }
                if(parameters.empty())
                    std::cout << "No parameters exist\n";
            }
        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "add"));


        slot = new mo::TSlot<void(std::string)>(std::bind([&current_node, &current_stream, &current_param](std::string value)->void
        {
            if(current_param && current_node && current_param->checkFlags(mo::Input_e))
            {
                auto token_index = value.find(':');
                if(token_index != std::string::npos)
                {
                    auto stream = current_node->getDataStream();
                    auto space_index = value.substr(token_index+1).find(' ');
                    std::string output_name;
                    mo::ParamType flags = mo::BlockingStreamBuffer_e;
                    if(space_index != std::string::npos)
                    {
                        output_name = value.substr(token_index + 1, space_index);
                        std::string buffer_type = value.substr(token_index + space_index + 2);
                        flags = mo::ParamType(mo::stringToParamType(buffer_type) | mo::ForceBufferedConnection_e);
                    }else
                    {
                        output_name = value.substr(token_index + 1);
                    }

                    auto output_node = stream->getNode(value.substr(0, token_index));
                    if(output_node)
                    {
                        auto output_param = output_node->getOutput(output_name);
                        if(output_param)
                        {
                            auto input_param = dynamic_cast<mo::InputParam*>(current_param);
                            if(input_param)
                            {
                                if(current_node->connectInput(output_node, output_param, input_param, flags))
                                {
                                    std::cout << "Successfully set input of " << current_param->getName() << " to " << output_param->getName() << "\n";
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            if(!current_param)
            {

            }else
            {
                auto func = mo::SerializationFactory::instance()->getTextDeSerializationFunction(current_param->getTypeInfo());
                if(func)
                {
                    std::stringstream ss;
                    ss << value;
                    mo::Mutex_t::scoped_lock lock(current_param->mtx());
                    if(func(current_param, ss))
                        std::cout << "Successfully set " << current_param->getTreeName() << " to " << value << std::endl;
                    else
                        std::cout << "Failed to set " << current_param->getTreeName() << " to " << value << std::endl;
                    return;
                }else
                {
                    std::cout << "No text deserialization function found for " << current_param->getTypeInfo().name() << std::endl;
                }
            }
            if (current_node)
            {
                auto params = current_node->getAllParams();
                for(auto& param : params)
                {
                    auto pos = value.find(param->getName());
                    if(pos != std::string::npos && value.size() > param->getName().size() + 1 && value[param->getName().size()] == ' ')
                    {
                        //std::cout << "Setting value for parameter " << param->getName() << " to " << value.substr(pos + param->getName().size() + 1) << std::endl;
                        //std::stringstream ss;
                        //ss << value.substr(pos + param->getName().size() + 1);
                        //Parameters::Persistence::Text::DeSerialize(&ss, param);
                        return;
                    }
                }
                std::cout << "Unable to find parameter by name for set string: " << value << std::endl;
                auto pos = value.find("sync");
                if(pos == 0)
                {
                    pos = value.find(' ');
                    current_node->setSyncInput(value.substr(pos + 1));
                }

            }else if(current_stream)
            {
            }
            std::cout << "Unable to set value to " << value << std::endl;
        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "set"));
        slot = new mo::TSlot<void(std::string)>(std::bind([&current_node](std::string name)
        {
            if(current_node)
            {
                current_node->setTreeName(name);
            }
        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "rename"));

        slot = new mo::TSlot<void(std::string)>(std::bind([&current_node, &current_stream](std::string name)
        {
            mo::RelayManager* mgr = nullptr;
            if (current_node)
            {
                mgr = current_node->getDataStream()->getRelayManager();
            }
            if (current_stream)
            {
                mgr = current_stream->getRelayManager();
            }
            std::vector<std::shared_ptr<mo::ISignalRelay>> relays;
            if (mgr)
            {
                relays = mgr->getRelays(name);
            }
            auto table = PerModuleInterface::GetInstance()->GetSystemTable();
            if (table)
            {
                auto global_signal_manager = table->getSingleton<mo::RelayManager>();
                if(global_signal_manager)
                {
                    auto global_relays = global_signal_manager->getRelays(name);
                    relays.insert(relays.end(), global_relays.begin(), global_relays.end());
                }
            }
            if (relays.size() == 0)
            {
                std::cout << "No signals found with name: " << name << std::endl;
                return;
            }
            int idx = 0;
            mo::ISignalRelay* relay = nullptr;
            if (relays.size() > 1)
            {
                for (auto relay_ : relays)
                {
                    std::cout << idx << " - " << relay_->getSignature().name();
                    ++idx;
                }
                std::cin >> idx;
                if(idx >= 0 && idx < relays.size())
                    relay = relays[idx].get();
            }else if(relays.size() == 1)
            {
                relay = relays[0].get();
            }

            mo::TSignalRelay<void(void)>* typed = dynamic_cast<mo::TSignalRelay<void(void)>*>(relay);
            if(typed)
            {
                (*typed)();
                return;
            }
            THROW(debug) << "Signal serialization needs to be reimplemented";
            /*auto proxy = Signals::serialization::text::factory::instance()->get_proxy(signals[idx]);
            if(proxy)
            {
                proxy->send(signals[idx], "");
                delete proxy;
            }*/
        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "emit"));

#ifdef HAVE_WT
        boost::thread web_thread;
        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&current_stream, &web_thread, argc, argv](std::string null)->void
        {
            if(current_stream)
            {
                rcc::shared_ptr<vclick::WebSink> sink = current_stream->getNode("WebSink0");
                if(!sink)
                {
                    sink = rcc::shared_ptr<vclick::WebSink>::create();
                    current_stream->addNode(sink);
                    auto fg = current_stream->getNode("frame_grabber_openni20");
                    sink->connectInput(fg, fg->getParam("current_frame"), sink->getInput("point_cloud"));

                    auto foreground_estimator = current_stream->getNode("ForegroundEstimate0");
                    sink->connectInput(foreground_estimator, foreground_estimator->getParam("point_mask"),
                        sink->getInput("foreground_mask"));

                    sink->connectInput(foreground_estimator, foreground_estimator->getParam("background_model"),
                        sink->getInput("background_model"));
                }


                web_thread = boost::thread(std::bind(
                    [argc, argv, sink]()->void
                {
                    vclick::WebUi::StartServer(argc, argv, sink);
                }));
            }
        }, std::placeholders::_1));

        connections.push_back(manager.connect(slot, "web-ui"));
#endif
        slot = new mo::TSlot<void(std::string)>(std::bind(
            [](std::string level)
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
        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "log"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
        [](std::string directory)
        {
            int idx = 0;
            if(auto pos = directory.find(',') != std::string::npos)
            {
                idx = boost::lexical_cast<int>(directory.substr(0, pos));
                directory = directory.substr(pos + 1);
            }
            mo::MetaObjectFactory::instance()->getObjectSystem()->AddLibraryDir(directory.c_str(), idx);
        }, std::placeholders::_1));

        connections.push_back(manager.connect(slot, "link"));

        slot = new mo::TSlot<void(std::string)>(std::bind([](std::string ms)
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(boost::lexical_cast<int>(ms)));
        }, std::placeholders::_1));

        connections.push_back(manager.connect(slot, "wait"));


        std::vector<std::string> command_list;
        slot = new mo::TSlot<void(std::string)>(std::bind([&command_list](std::string filename)
        {
            std::ifstream ifs(filename);
            if(ifs.is_open())
            {
                std::string line;
                while(std::getline(ifs, line))
                {
                    if(line[line.size() - 1] == '\n' || line[line.size() - 1] == '\r')
                        line = line.substr(0, line.size() - 1);
                    command_list.push_back(line);
                }
                if(command_list.size())
                    std::reverse(command_list.begin(), command_list.end());
            }else
            {
                LOG(warning) << "Unable to load scripting file: " << filename;
            }

        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "run"));
        if (vm.count("config"))
        {
            std::stringstream ss;
            ss << "load " << vm["config"].as<std::string>();
            command_list.emplace_back(ss.str());
        }
        if(vm.count("launch"))
        {
            std::stringstream ss;
            ss << "load " << vm["launch"].as<std::string>();
            command_list.emplace_back(ss.str());
        }

        if (vm.count("file"))
        {
            auto relay = manager.getRelay<void(std::string)>("load_file");
            if(relay)
            {
                std::string file = vm["file"].as<std::string>();
                (*relay)(file);
            }
        }

        print_options();
        bool compiling = false;
        bool rcc_enabled = !vm["disable-rcc"].as<bool>();
        if(rcc_enabled)
            mo::MetaObjectFactory::instance()->checkCompile();
        auto compile_check_function = [&_dataStreams, &compiling, rcc_enabled]()
        {
            if(rcc_enabled)
            {
                if (mo::MetaObjectFactory::instance()->checkCompile())
                {
                    std::cout << "Recompiling...\n";
                    for (auto& ds : _dataStreams)
                    {
                        ds->stopThread();
                    }
                    compiling = true;
                }
                if (compiling)
                {
                    if (!mo::MetaObjectFactory::instance()->isCompileComplete())
                    {
                        std::cout << "Still compiling\n";
                    }
                    else
                    {
                        for (auto& ds : _dataStreams)
                        {
                            ds->stopThread();
                        }
                        if (mo::MetaObjectFactory::instance()->swapObjects())
                        {
                            std::cout << "Object swap success\n";
                            for (auto& ds : _dataStreams)
                            {
                                ds->startThread();
                            }
                        }
                        else
                        {
                            std::cout << "Failed to recompile\n";
                        }
                        compiling = false;
                    }
                }
            }
        };

        slot = new mo::TSlot<void(std::string)>(std::bind([&rcc_enabled](std::string value)
        {
            rcc_enabled = boost::lexical_cast<bool>(value);
        }, std::placeholders::_1));
        connections.push_back(manager.connect(slot, "rcc"));

        bool disable_input = vm["disable-input"].as<bool>();
        auto io_func = [&command_list, &manager, &print_options, disable_input]()
        {
            std::string command_line;
            bool skip = (command_list.size() == 0) && disable_input;
            if (command_list.size())
            {
                command_line = command_list.back();
                command_list.pop_back();
            }
            else
            {
                if(!disable_input)
                    if(std::cin.peek())
                        std::getline(std::cin, command_line);
            }
            if(!skip)
            {
                std::stringstream ss;
                ss << command_line;
                std::string command;
                std::getline(ss, command, ' ');
                auto relay = manager.getRelay<void(std::string)>(command);
                if (relay)
                {
                    std::string rest;
                    std::getline(ss, rest);
                    try
                    {
                        LOG(debug) << "Running command (" << command << ") with arguments: " << rest;
                        (*relay)(rest);
                    }
                    catch(std::exception& e)
                    {
                        LOG(warning) << "Executing command (" << command << ") with arguments: " << rest << " failed due to: "
                                     << "[" << typeid(e).name() << "] - " << e.what();
                    }
                    catch (...)
                    {
                        LOG(warning) << "Executing command (" << command << ") with arguments: " << rest << " failed miserably";
                    }
                }
                else
                {
                    if(command_line.size())
                    {
                        LOG(warning) << "Invalid command: " << command_line;
                        print_options();
                    }
                }

            }

            for (int i = 0; i < 20; ++i)
                mo::ThreadSpecificQueue::runOnce();
        };

        if(vm.count("script"))
        {
            auto relay = manager.getRelay<void(std::string)>("run");
            if(relay)
            {
                std::string file = vm["script"].as<std::string>();
                (*relay)(file);
            }
        }
        int run_time = -1;
        if(vm.count("profile-for") != 0)
        {
            run_time = vm["profile-for"].as<int>();
        }
        boost::thread io_thread = boost::thread(std::bind(
        [&io_func, &_dataStreams, run_time]()
        {
            auto start = boost::posix_time::microsec_clock::universal_time();
            while(!quit)
            {
                io_func();
                if(run_time != -1)
                {
                    auto now = boost::posix_time::microsec_clock::universal_time();
                    if(boost::posix_time::time_duration(now - start).total_seconds() > run_time)
                    {
                        quit = true;
                    }
                }
            }

            std::cout << "IO thread shutting down\n";
        }));
        boost::posix_time::ptime last_compile_check_time = boost::posix_time::microsec_clock::universal_time();

        signal(SIGINT, sig_handler);
        while(!quit)
        {
            auto current_time = boost::posix_time::microsec_clock::universal_time();
            if (boost::posix_time::time_duration(current_time - last_compile_check_time).total_milliseconds() > 1000)
            {
                last_compile_check_time = current_time;
                compile_check_function();
            }else
            {
                boost::this_thread::sleep_for(boost::chrono::seconds(1));
            }
        }
        io_thread.~thread();
        gui_thread.interrupt();
        gui_thread.join();
        for (auto& ds : _dataStreams)
        {
            ds->stopThread();
        }
        mo::ThreadSpecificQueue::cleanup();
        _dataStreams.clear();
        LOG(info) << "Gui thread shut down complete";
        mo::ThreadPool::Instance()->Cleanup();
        LOG(info) << "Thread pool cleanup complete";
        delete g_allocator;
        mo::Allocator::cleanupThreadSpecificAllocator();
        return 0;
    }
    gui_thread.interrupt();
    gui_thread.join();

    LOG(info) << "Gui thread shut down complete, cleaning up thread pool";
    mo::ThreadPool::Instance()->Cleanup();
    LOG(info) << "Thread pool cleanup complete";
    delete g_allocator;
    return 0;
}
