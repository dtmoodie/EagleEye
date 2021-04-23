#include <Aquila/core.hpp>
#include <Aquila/core/IGraph.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
//#include <Aquila/gui.hpp>
//#include <Aquila/gui/UiCallbackHandlers.h>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeFactory.hpp>
#include <MetaObject/logging/logging.hpp>

#include <MetaObject/core/detail/Allocator.hpp>

#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/RelayManager.hpp>

#include <MetaObject/thread/ThreadPool.hpp>

#include <RuntimeObjectSystem/RuntimeObjectSystem.h>

#include <boost/asio.hpp>
#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/core.hpp>
#include <boost/log/exceptions.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/version.hpp>
#include <signal.h> // SIGINT, etc

#include "MetaObject/MetaParameters.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <fstream>

std::string printParam(mo::IParam* param)
{
    std::stringstream ss;
    ss << " - " << param->getTreeName() << " ";
    ;
    if (param->checkFlags(mo::ParamFlags::kINPUT))
    {
        if (auto input = dynamic_cast<mo::ISubscriber*>(param))
        {
            ss << " [";
            auto input_param = input->getPublisher();
            if (input_param)
            {
                input_param->print(ss);
                ss << input_param->getTreeName() << " ";
                auto hdr = input_param->getNewestHeader();

                if (hdr)
                {
                    ss << " " << *hdr;
                }
            }
            else
            {
                ss << "input not set";
            }
            ss << "]\n";
        }
    }
    else
    {
        param->print(ss);
    }
    return ss.str();
}

void PrintNodeTree(aq::nodes::INode* node, int depth)
{
    if (!node)
        return;
    for (int i = 0; i < depth; ++i)
    {
        std::cout << "=";
    }
    std::cout << node->getName() << std::endl;
    auto children = node->getChildren();
    for (size_t i = 0; i < children.size(); ++i)
    {
        PrintNodeTree(children[i].get(), depth + 1);
    }
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& strs)
{
    for (const auto& str : strs)
    {
        os << str << "\n ";
    }
    return os;
}

void PrintBuffers(aq::nodes::INode* node, std::vector<std::string>& printed_nodes)
{
    std::string name = node->getName();
    if (std::find(printed_nodes.begin(), printed_nodes.end(), name) != printed_nodes.end())
    {
        return;
    }
    printed_nodes.push_back(name);
    std::vector<mo::ISubscriber*> inputs = node->getInputs();
    std::cout << "--------\n" << name << std::endl;
    for (mo::ISubscriber* input : inputs)
    {
        mo::IParam* param = input->getInputParam();
        mo::Buffer::IBuffer* buf = dynamic_cast<mo::Buffer::IBuffer*>(param);
        if (buf)
        {
            std::cout << param->getTreeName() << " - " << buf->getSize() << std::endl;
        }
    }

    auto children = node->getChildren();
    for (auto child : children)
    {
        PrintBuffers(child.get(), printed_nodes);
    }
}

void printStatus(aq::nodes::INode* node, std::vector<std::string>& printed_nodes)
{
    std::string name = node->getName();
    if (std::find(printed_nodes.begin(), printed_nodes.end(), name) != printed_nodes.end())
    {
        return;
    }
    printed_nodes.push_back(name);
    std::cout << "--------\n" << name << std::endl;
    if (node->getMutex().try_lock())
    {
        std::cout << "Locked: unlocked. Modified: " << node->getModified() << std::endl;
        node->getMutex().unlock();
    }
    else
    {
        std::cout << "Locked: locked. Modified: " << node->getModified() << std::endl;
    }

    auto children = node->getChildren();
    for (auto child : children)
    {
        printStatus(child.get(), printed_nodes);
    }
}

static volatile bool quit;

void sig_handler(int s)
{
    switch (s)
    {
    case SIGSEGV: {
        // std::cout << "Caught SIGSEGV " << mo::print_callstack(2, true);
        break;
    }
    case SIGINT: {
        // std::cout << "Caught SIGINT " << mo::print_callstack(2, true);
        std::cout << "Caught SIGINT, shutting down" << std::endl;
        static int count = 0;
        quit = true;
        ++count;
        if (count > 2)
        {
            std::terminate();
        }
        return;
    }
    case SIGILL: {
        std::cout << "Caught SIGILL " << std::endl;
        break;
    }
    case SIGTERM: {
        std::cout << "Caught SIGTERM " << std::endl;
        break;
    }
#ifndef _MSC_VER
    case SIGKILL: {
        std::cout << "Caught SIGKILL " << std::endl;
        break;
    }
#endif
    default: {
        std::cout << "Caught signal " << s << std::endl;
    }
    }
}

int main(int argc, char* argv[])
{
    BOOST_LOG_TRIVIAL(info) << "Initializing";
    boost::program_options::options_description desc("Allowed options");
    auto table = SystemTable::instance();
    mo::MetaObjectFactory::Ptr_t factory = mo::MetaObjectFactory::instance((table.get());
    // clang-format off
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
            ("preset", boost::program_options::value<std::string>()->default_value("Default"), "Preset config file setting");
    // clang-format on
    boost::program_options::variables_map vm;

    auto parsed_options =
        boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    boost::program_options::store(parsed_options, vm);
    mo::initMetaParamsModule();

    factory->registerTranslationUnit();
    if (vm.count("log-dir"))
        aq::core::initModule(&factory, vm["log-dir"].as<std::string>());
    else
        aq::core::initModule(&factory);
    aq::gui::initModule(&factory);

    auto g_allocator = mo::Allocator::createAllocator();
    mo::CvAllocatorProxy<mo::CUDA> gpu_allocator(g_allocator);
    mo::CvAllocatorProxy<mo::CPU> cpu_allocator(g_allocator);
    cv::cuda::GpuMat::setDefaultAllocator(&gpu_allocator);
    cv::Mat::setDefaultAllocator(&cpu_allocator);

    g_allocator->setName("Global Allocator");

    auto unrecognized = boost::program_options::collect_unrecognized(parsed_options.options,
                                                                     boost::program_options::include_positional);
    std::map<std::string, std::string> replace_map;
    std::map<std::string, std::string> variable_replace_map;
    std::stringstream currentDate;
    boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
    currentDate << timeLocal.date().year() << "-" << std::setfill('0') << std::setw(2)
                << timeLocal.date().month().as_number() << "-" << std::setfill('0') << std::setw(2)
                << timeLocal.date().day().as_number();
    replace_map["${date}"] = currentDate.str();
    replace_map["${hostname}"] = boost::asio::ip::host_name();
    currentDate.str(std::string());
    currentDate << std::setfill('0') << std::setw(2) << timeLocal.time_of_day().hours() << std::setfill('0')
                << std::setw(2) << timeLocal.time_of_day().minutes();
    replace_map["${hour}"] = currentDate.str();
    replace_map["${pid}"] = boost::lexical_cast<std::string>(boost::log::aux::this_process::get_id());
    if (vm.count("config"))
    {
        replace_map["${config_file_dir}"] =
            boost::filesystem::path(vm["config"].as<std::string>()).parent_path().string();
    }
    for (auto& option : unrecognized)
    {
        auto pos = option.find(":=");
        if (pos != std::string::npos)
        {
            std::string start = option.substr(0, pos);
            std::string end = option.substr(pos + 2);
            if (end[0] == '\"' && end[end.size() - 1] == '\"')
            {
                end = end.substr(1, end.size() - 2);
            }
            replace_map["${" + start + "}"] = end;
            continue;
        }
        pos = option.find("=");
        if (option.find("--") == 0 && pos != std::string::npos)
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
    if (replace_map.size())
    {
        std::stringstream ss;
        for (const auto& pair : replace_map)
            ss << "\n" << pair.first << " = " << pair.second;
        MO_LOG(debug, "Input string replacements: {}", ss.str());
    }

    if (variable_replace_map.size())
    {
        std::stringstream ss;
        for (const auto& pair : variable_replace_map)
            ss << "\n" << pair.first << " = " << pair.second;
        MO_LOG(debug, "Input variable replacements: ", ss.str());
    }

    if (vm["profile"].as<bool>() || vm.count("profile-for"))
    {
        mo::initProfiling();
    }
    cv::cuda::setDevice(vm["gpu"].as<int>());
    {
        boost::posix_time::ptime initialization_start = boost::posix_time::microsec_clock::universal_time();
        MO_LOG(info, "Initializing GPU...");
        cv::cuda::GpuMat(10, 10, CV_32F);
        boost::posix_time::ptime initialization_end = boost::posix_time::microsec_clock::universal_time();
        if (boost::posix_time::time_duration(initialization_end - initialization_start).total_seconds() > 1)
        {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, cv::cuda::getDevice());
            MO_LOG(warning,
                   "Initialization took {} ms. CUDA code likely not generated for this architecture ({}.{})",
                   boost::posix_time::time_duration(initialization_end - initialization_start).total_milliseconds(),
                   props.major,
                   props.minor);
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
    }
    else
    {
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
    }
    boost::filesystem::path currentDir = boost::filesystem::path(argv[0]).parent_path();
#ifdef _MSC_VER
#ifdef _DEBUG
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_guid.dll");
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_cored.dll");
#else
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_gui.dll");
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_core.dll");
#endif
#else
#ifdef NDEBUG
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_gui.so");
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_core.so");
#else
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_guid.so");
    mo::MetaObjectFactory::instance()->loadPlugin("aquila_cored.so");
#endif
#endif

    currentDir = boost::filesystem::path(currentDir.string() + "/Plugins");

    MO_LOG(info) << "Looking for plugins in: " << currentDir.string();
    boost::filesystem::directory_iterator end_itr;
    if (boost::filesystem::is_directory(currentDir))
    {
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
    boost::thread gui_thread([] {
        mo::ThreadRegistry::instance()->registerThread(mo::ThreadRegistry::GUI);
        boost::mutex dummy_mtx; // needed for cv
        boost::condition_variable cv;
        auto notifier = mo::ThreadSpecificQueue::registerNotifier([&cv]() { cv.notify_all(); });
        while (!boost::this_thread::interruption_requested())
        {
            mo::setThreadName("SimpleConsole GUI thread");
            try
            {
                boost::mutex::scoped_lock lock(dummy_mtx);
                cv.wait(lock);
                mo::ThreadSpecificQueue::run();
            }
            catch (boost::thread_interrupted& err)
            {
                (void)err;
                break;
            }
            catch (mo::ExceptionWithCallStack<cv::Exception>& e)
            {
                MO_LOG(debug, "Opencv exception with callstack {} {}", e.what(), e.callStack());
            }
            catch (mo::IExceptionWithCallStackBase& e)
            {
                MO_LOG(debug) << "Exception with callstack " << e.callStack();
            }
            catch (cv::Exception& e)
            {
                MO_LOG(debug) << "OpenCV exception: " << e.what();
            }
            catch (...)
            {
                MO_LOG(debug) << "Unknown / unhandled exception thrown in gui thread event handler";
            }
            try
            {
                // cv::waitKey(1);
                aq::WindowCallbackHandler::EventLoop::Instance()->run();
            }
            catch (mo::ExceptionWithCallStack<cv::Exception>& e)
            {
                (void)e;
            }
            catch (cv::Exception& e)
            {
                (void)e;
            }
            catch (boost::thread_interrupted& /*err*/)
            {
                break;
            }
            catch (...)
            {
            }
        }
        mo::ThreadSpecificQueue::cleanup();
        MO_LOG(info) << "Gui thread shutting down naturally";
    });
    mo::setThreadName(gui_thread, "Gui-thread");
    mo::RelayManager manager;

    if (vm.count("plugins"))
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
                    mo::MetaObjectFactory::instance().loadPlugin(file);
                }
            }
        }
    }

    if (vm["mode"].as<std::string>() == "batch") {}
    else
    {
        std::vector<rcc::shared_ptr<aq::IGraph>> _Graphs;
        rcc::weak_ptr<aq::IGraph> current_stream;
        rcc::weak_ptr<aq::nodes::INode> current_node;
        mo::IParam* current_param = nullptr;

        auto print_options = []() -> void {
            std::cout << "- Options: \n"
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
        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&documents_list](std::string null) -> void {
                (void)null;
                documents_list.clear();
                auto constructors =
                    mo::MetaObjectFactory::instance().getConstructors(aq::nodes::IFrameGrabber::getHash());
                int index = 0;
                for (auto constructor : constructors)
                {
                    auto fg_info = dynamic_cast<aq::nodes::IFrameGrabber::InterfaceInfo*>(constructor->GetObjectInfo());
                    if (fg_info)
                    {
                        auto documents = fg_info->listLoadablePaths();
                        for (auto& document : documents)
                        {
                            std::cout << " - " << index << "  [" << fg_info->GetObjectName() << "] " << document
                                      << "\n";
                            documents_list.emplace_back(document, fg_info->getDisplayName());
                            ++index;
                        }
                    }
                }
            },
            std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "list_devices"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&_Graphs, &documents_list](std::string doc) -> void {
                std::string fg_override;
                int index = -1;
#ifdef BOOST_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP
                if (!boost::conversion::detail::try_lexical_convert(doc, index))
                {
                    index = -1;
                }
#else
                try
                {
                    index = boost::lexical_cast<int>(doc);
                }
                catch (...)
                {
                    index = -1;
                }
#endif
                if (index != -1 && index >= 0 && static_cast<size_t>(index) < documents_list.size())
                {
                    doc = documents_list[static_cast<size_t>(index)].first;
                    fg_override = documents_list[static_cast<size_t>(index)].second;
                }
                auto ds = aq::IGraph::create(doc, fg_override);
                if (ds)
                {
                    ds->startThread();
                    _Graphs.push_back(ds);
                }
            },
            std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "load_file"));

        slot = new mo::TSlot<void(std::string)>(
            std::bind([](std::string) -> void { quit = true; }, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "quit"));

        auto func = [&_Graphs, &current_stream, &current_node, &current_param](std::string what) -> void {
            if (what == "streams")
            {
                for (auto& itr : _Graphs)
                {
                    auto fgs = itr->getTopLevelNodes();
                    for (auto& fg : fgs)
                    {
                        if (auto frame_grabber = fg.DynamicCast<aq::nodes::IFrameGrabber>())
                        {
                            std::cout << " - " << frame_grabber->GetPerTypeId() << " - "
                                      << frame_grabber->loaded_document << "\n";
                        }
                        else
                        {
                        }
                    }
                }
                if (_Graphs.empty())
                    std::cout << "No streams exist\n";
            }
            if (what == "nodes")
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
                    PrintNodeTree(current_node.get(), 0);
                }
            }
            if (what == "parameters" || what == "params")
            {
                std::vector<mo::IParam*> parameters;
                if (current_node)
                {
                    parameters = current_node->getParams();
                }
                if (current_stream) {}
                for (auto& itr : parameters)
                {
                    try
                    {
                        std::cout << printParam(itr) << std::endl;
                    }
                    catch (...)
                    {
                    }
                }
                if (parameters.empty())
                    std::cout << "No parameters exist\n";
            }
            if (what == "current" || what.empty())
            {
                if (current_stream)
                {
                    auto fgs = current_stream->getTopLevelNodes();
                    for (auto& fg : fgs)
                    {
                        if (auto f_g = fg.DynamicCast<aq::nodes::IFrameGrabber>())
                        {
                            std::cout << " - Datasource: " << f_g->loaded_document << "\n";
                        }
                    }
                }
                if (current_node)
                {
                    std::cout << " - Current node: " << current_node->getTreeName() << "\n";
                    std::cout << "    Type: " << current_node->GetTypeName() << std::endl;
                }
                if (current_param)
                {
                    std::cout << " - Current parameter: " << current_param->getTreeName() << "\n";
                }
                if (!current_node && !current_param && !current_stream)
                    std::cout << "Nothing currently selected\n";
            }
            if (what == "signals")
            {
                if (current_node)
                {
                    std::vector<mo::SignalInfo*> infos;
                    current_node->getSignalInfo(infos);
                    for (auto& info : infos)
                    {
                        std::cout << info->print();
                    }
                }
                if (current_stream)
                {
                    // current_stream->GetSignalManager()->print_signal_map();
                }
            }
            if (what == "inputs")
            {
                if (current_param && current_node)
                {
                    auto potential_inputs =
                        current_node->getGraph()->getVariableManager()->getOutputParams(current_param->getTypeInfo());
                    std::stringstream ss;
                    if (potential_inputs.size())
                    {
                        ss << "Potential inputs: \n";
                        for (auto& input : potential_inputs)
                        {
                            ss << " - " << input->getTreeName() << "\n";
                        }
                        std::cout << ss.str() << std::endl;
                        return;
                    }
                    std::cout << "Unable to find any matching inputs for variable with name: "
                              << current_param->getName() << " with type: " << current_param->getTypeInfo().name()
                              << std::endl;
                }
                if (current_node)
                {
                    auto params = current_node->getParams();
                    std::stringstream ss;
                    for (auto param : params)
                    {
                        if (param->checkFlags(mo::ParamFlags::Input_e))
                        {
                            ss << " -- " << param->getTreeName() << " [ " << param->getTypeInfo().name() << " ]\n";
                            auto potential_inputs =
                                current_node->getGraph()->getVariableManager()->getOutputParams(param->getTypeInfo());
                            for (auto& input : potential_inputs)
                            {
                                ss << " - " << input->getTreeName();
                            }
                        }
                    }
                    std::cout << ss.str() << std::endl;
                }
            }
            if (what == "projects")
            {
                THROW(debug, "Needs to be reimplemented");
            }
            if (what == "plugins")
            {
                auto plugins = mo::MetaObjectFactory::instance().listLoadedPlugins();
                std::stringstream ss;
                ss << "\n";
                for (auto& plugin : plugins)
                {
                    ss << plugin << "\n";
                }
                if (plugins.empty())
                    ss << "No plugins loaded\n";
                std::cout << ss.str() << std::endl;
            }
            if (what == "tree")
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
                    auto nodes = current_node->getGraph()->getNodes();
                    for (auto node : nodes)
                        PrintNodeTree(node.get(), 0);
                }
            }
            if (what == "buffers")
            {
                if (current_stream)
                {
                    auto nodes = current_stream->getNodes();
                    std::vector<std::string> printed;
                    for (auto node : nodes)
                    {
                        PrintBuffers(node.get(), printed);
                    }
                }
            }
            if (what == "status")
            {
                if (current_stream)
                {
                    std::cout << "Graph modified: " << current_stream->getDirty() << std::endl;
                    auto nodes = current_stream->getNodes();
                    std::vector<std::string> printed;
                    for (auto node : nodes)
                    {
                        printStatus(node.get(), printed);
                    }
                }
            }
        };
        slot = new mo::TSlot<void(std::string)>(std::bind(func, std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "print"));
        connections.push_back(manager.connect(slot, "ls"));
        slot = new mo::TSlot<void(std::string)>(std::bind(
            [](std::string obj) {
                auto pos = obj.find(' ');
                IObjectInfo::Verbosity verb = IObjectInfo::INFO;
                if (pos != std::string::npos)
                {
                    if (obj.substr(pos + 1) == "DEBUG")
                        verb = IObjectInfo::DEBUG;
                    if (obj.substr(pos + 1) == "RCC")
                        verb = IObjectInfo::RCC;
                    obj = obj.substr(0, pos);
                }
                IObjectConstructor* constructor = mo::MetaObjectFactory::instance().getConstructor(obj.c_str());
                if (constructor)
                {
                    mo::IMetaObjectInfo* info = dynamic_cast<mo::IMetaObjectInfo*>(constructor->GetObjectInfo());
                    if (info)
                    {
                        std::cout << info->Print(verb);
                    }
                }
                else
                {
                    std::cout << "No constructor found for " << obj;
                }
            },
            std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "info"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&current_stream, &current_node, &variable_replace_map, &replace_map](std::string file) {
                if (current_stream)
                {
                    // current_stream->SaveStream(file);
                    rcc::shared_ptr<aq::IGraph> stream(current_stream);
                    std::vector<rcc::shared_ptr<aq::IGraph>> streams;
                    streams.push_back(stream);
                    aq::IGraph::save(file, streams, variable_replace_map, replace_map);
                    stream->startThread();
                }
                else if (current_node)
                {
                }
            },
            std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "save"));

        bool quit_on_eos = vm["quit-on-eos"].as<bool>();
        mo::TSlot<void()> eos_slot(std::bind([]() {
            MO_LOG_FIRST_N(info, 1) << "End Of Stream received, shutting down";
            quit = true;
        }));

        std::vector<std::shared_ptr<mo::Connection>> eos_connections;
        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&_Graphs,
             &current_stream,
             &current_node,
             quit_on_eos,
             &eos_connections,
             &eos_slot,
             &variable_replace_map,
             &replace_map](std::string file) {
                auto pos = file.find(' ');
                std::string preset = "Default";
                if (pos != std::string::npos)
                {
                    preset = file.substr(pos + 1);
                    MO_LOG(info) << "Using preset '" << preset << "'";
                    file = file.substr(0, pos);
                }
                replace_map["${config_file_dir}"] = boost::filesystem::path(file).parent_path().string();
                auto streams = aq::IGraph::load(file, variable_replace_map, replace_map, preset);
                if (streams.size())
                {
                    for (auto& stream : streams)
                    {
                        stream->startThread();
                        _Graphs.push_back(stream);
                        if (quit_on_eos)
                        {
                            stream->getRelayManager()->connect(&eos_slot, "eos");
                        }
                    }
                    std::cout << "Load of " << file << " complete" << std::endl;
                    ;
                }
                else
                {
                    std::cout << "Load of " << file << " failed" << std::endl;
                    ;
                }
            },
            std::placeholders::_1));
        _slots.emplace_back(slot);
        connections.push_back(manager.connect(slot, "load"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&_Graphs, &current_stream, &current_node, &current_param](std::string what) {
                if (what == "null")
                {
                    current_stream.reset();
                    current_node.reset();
                    current_param = nullptr;
                }
                int idx = -1;
                std::string name;

#ifdef BOOST_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP
                if (!boost::conversion::detail::try_lexical_convert(what, idx))
                {
                    idx = -1;
                }
#else
                try
                {
                    idx = boost::lexical_cast<int>(what);
                }
                catch (...)
                {
                    idx = -1;
                }
#endif

                if (idx == -1)
                {
                    name = what;
                }
                if (idx != -1)
                {
                    std::cout << "Selecting stream " << idx << std::endl;
                    for (auto& itr : _Graphs)
                    {
                        if (itr != nullptr)
                        {
                            if (itr->GetPerTypeId() == idx)
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
                if (current_stream)
                {
                    // look for a node with this name, relative then look absolute
                    auto node = current_stream->getNode(what);
                    if (node)
                    {
                        current_node = node;
                    }
                    std::cout << "No node found with given name " << what << std::endl;
                }
                if (current_node)
                {
                    auto child = current_node->getChild(what);
                    if (child)
                    {
                        current_node = child.get();
                        current_stream.reset();
                        current_param = nullptr;
                        std::cout << "Successfully set node to " << child->getTreeName() << "\n";
                        std::vector<mo::IParam*> parameters;
                        if (current_node)
                        {
                            parameters = current_node->getParams();
                        }
                        if (current_stream) {}
                        for (auto& itr : parameters)
                        {
                            std::stringstream ss;
                            try
                            {
                                if (itr->checkFlags(mo::ParamFlags::Input_e))
                                {
                                    if (auto input = dynamic_cast<mo::InputParam*>(itr))
                                    {
                                        std::stringstream ss;
                                        ss << " - " << itr->getTreeName() << " [";
                                        auto input_param = input->getInputParam();
                                        if (input_param)
                                        {
                                            ss << input_param->getTreeName();
                                        }
                                        else
                                        {
                                            ss << "input not set";
                                        }
                                        ss << "]\n";
                                        std::cout << ss.str();
                                    }
                                }
                                else
                                {
                                    auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(
                                        itr->getTypeInfo());
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
                            }
                            catch (...)
                            {
                                // std::cout << " - " << itr->getTreeName() << "\n";
                            }
                        }
                        if (parameters.empty())
                            std::cout << "No parameters exist\n";
                        return;
                    }
                    else
                    {
                        auto stream = current_node->getGraph();
                        if (auto node = stream->getNode(what))
                        {
                            current_node = node;
                            current_stream.reset();
                            current_param = nullptr;
                            std::cout << "Successfully set node to " << node->getTreeName() << "\n";
                            std::vector<mo::IParam*> parameters;
                            if (current_node)
                            {
                                parameters = current_node->getParams();
                            }
                            if (current_stream) {}
                            for (auto& itr : parameters)
                            {
                                std::cout << printParam(itr) << std::endl;
                            }
                            if (parameters.empty())
                                std::cout << "No parameters exist\n";
                            return;
                        }
                        else
                        {
                            auto params = current_node->getParams();
                            for (auto& param : params)
                            {
                                std::string name = param->getName();
                                auto pos = name.find(':');
                                if (pos == std::string::npos)
                                {
                                    if (name == what)
                                    {
                                        current_param = param;
                                        current_stream.reset();
                                        return;
                                    }
                                }
                                else
                                {
                                    if (name.substr(pos + 1) == what)
                                    {
                                        current_param = param;
                                        current_stream.reset();
                                        return;
                                    }
                                }
                            }
                            for (auto& param : params)
                            {
                                if (param->getName().find(what) != std::string::npos)
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
            },
            std::placeholders::_1));

        connections.push_back(manager.connect(slot, "select"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&_Graphs, &current_stream, &current_node](std::string what) {
                (void)what;
                if (current_stream)
                {
                    auto itr = std::find(_Graphs.begin(), _Graphs.end(), current_stream.get());
                    if (itr != _Graphs.end())
                    {
                        _Graphs.erase(itr);
                        current_stream.reset();
                        std::cout << "Sucessfully deleted stream\n";
                        return;
                    }
                }
                else if (current_node)
                {
                    auto parents = current_node->getParents();
                    if (parents.size())
                    {
                        for (auto parent : parents)
                        {
                            parent->removeChild(current_node.get());
                        }
                        current_node.reset();
                        std::cout << "Sucessfully removed node from parent node\n";
                        return;
                    }
                    else if (auto stream = current_node->getGraph())
                    {
                        stream->removeNode(current_node.get());
                        current_node.reset();
                        std::cout << "Sucessfully removed node from Graph\n";
                        return;
                    }
                }
                std::cout << "Unable to delete item\n";
            },
            std::placeholders::_1));

        connections.push_back(manager.connect(slot, "delete"));

        slot = new mo::TSlot<void(std::string)>(
            std::bind([&print_options](std::string) -> void { print_options(); }, std::placeholders::_1));

        connections.push_back(manager.connect(slot, "help"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [](std::string filter) -> void {
                auto constructors = mo::MetaObjectFactory::instance().getConstructors();
                std::map<std::string, std::vector<IObjectConstructor*>> interface_map;
                for (auto constructor : constructors)
                {
                    IObjectInfo* info = constructor->GetObjectInfo();
                    if (info)
                    {
                        interface_map[info->GetInterfaceName()].push_back(constructor);
                    }
                }
                for (auto itr = interface_map.begin(); itr != interface_map.end(); ++itr)
                {
                    std::cout << "========= " << itr->first << std::endl;
                    for (auto ctr : itr->second)
                    {
                        std::string name = ctr->GetObjectInfo()->GetObjectName();
                        if (filter.size())
                        {
                            if (name.find(filter) == std::string::npos)
                            {
                                continue;
                            }
                        }
                        std::cout << "  " << name << std::endl;
                    }
                }
            },
            std::placeholders::_1));
        connections.push_back(manager.connect(slot, "list"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [](std::string verbosity) -> void {
                mo::MetaObjectFactory::PluginVerbosity verb = mo::MetaObjectFactory::brief;
                if (verbosity == "info")
                    verb = mo::MetaObjectFactory::info;
                if (verbosity == "debug")
                    verb = mo::MetaObjectFactory::debug;
                auto plugins = mo::MetaObjectFactory::instance().listLoadedPlugins(verb);
                std::stringstream ss;
                ss << "Loaded / failed plugins:\n";
                for (auto& plugin : plugins)
                {
                    ss << "  " << plugin << "\n";
                }
                std::cout << ss.str() << std::endl;
                ;
            },
            std::placeholders::_1));

        connections.push_back(manager.connect(slot, "plugins"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&current_node, &current_stream](std::string name) -> void {
                if (current_stream)
                {
                    auto added_nodes = current_stream->addNode(name);
                    if (added_nodes.size())
                        current_node = added_nodes[0];
                    std::vector<mo::IParam*> parameters;
                    if (current_node)
                    {
                        parameters = current_node->getParams();
                    }
                    for (auto& itr : parameters)
                    {
                        std::stringstream ss;
                        try
                        {
                            if (itr->checkFlags(mo::ParamFlags::Input_e))
                            {
                                if (auto input = dynamic_cast<mo::InputParam*>(itr))
                                {
                                    std::stringstream ss;
                                    ss << " - " << itr->getTreeName() << " [";
                                    auto input_param = input->getInputParam();
                                    if (input_param)
                                    {
                                        ss << input_param->getTreeName();
                                    }
                                    else
                                    {
                                        ss << "input not set";
                                    }
                                    ss << "]\n";
                                    std::cout << ss.str();
                                }
                            }
                            else
                            {
                                auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(
                                    itr->getTypeInfo());
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
                        }
                        catch (...)
                        {
                            // std::cout << " - " << itr->getTreeName() << "\n";
                        }
                    }
                    if (parameters.empty())
                        std::cout << "No parameters exist\n";
                    return;
                }
                if (current_node)
                {
                    auto added_nodes = aq::NodeFactory::Instance()->addNode(name, current_node.get());
                    if (added_nodes.size() == 1)
                    {
                        current_node = added_nodes[0];
                    }
                    std::vector<mo::IParam*> parameters;
                    if (current_node)
                    {
                        parameters = current_node->getParams();
                    }
                    for (auto& itr : parameters)
                    {
                        std::stringstream ss;
                        try
                        {
                            if (itr->checkFlags(mo::ParamFlags::Input_e))
                            {
                                if (auto input = dynamic_cast<mo::InputParam*>(itr))
                                {
                                    std::stringstream ss;
                                    ss << " - " << itr->getTreeName() << " [";
                                    auto input_param = input->getInputParam();
                                    if (input_param)
                                    {
                                        ss << input_param->getTreeName();
                                    }
                                    else
                                    {
                                        ss << "input not set";
                                    }
                                    ss << "]\n";
                                    std::cout << ss.str();
                                }
                            }
                            else
                            {
                                auto func = mo::SerializationFactory::instance()->getTextSerializationFunction(
                                    itr->getTypeInfo());
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
                        }
                        catch (...)
                        {
                            // std::cout << " - " << itr->getTreeName() << "\n";
                        }
                    }
                    if (parameters.empty())
                        std::cout << "No parameters exist\n";
                }
            },
            std::placeholders::_1));
        connections.push_back(manager.connect(slot, "add"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&current_node, &current_stream, &current_param](std::string value) -> void {
                if (current_param && current_node && current_param->checkFlags(mo::ParamFlags::Input_e))
                {
                    auto token_index = value.find(':');
                    if (token_index != std::string::npos)
                    {
                        auto stream = current_node->getGraph();
                        auto space_index = value.substr(token_index + 1).find(' ');
                        std::string output_name;
                        mo::ParamType flags = mo::BlockingStreamBuffer_e;
                        if (space_index != std::string::npos)
                        {
                            output_name = value.substr(token_index + 1, space_index);
                            std::string buffer_type = value.substr(token_index + space_index + 2);
                            flags = mo::ParamType(mo::stringToParamType(buffer_type) | mo::ForceBufferedConnection_e);
                        }
                        else
                        {
                            output_name = value.substr(token_index + 1);
                        }

                        auto output_node = stream->getNode(value.substr(0, token_index));
                        if (output_node)
                        {
                            auto output_param = output_node->getOutput(output_name);
                            if (output_param)
                            {
                                auto input_param = dynamic_cast<mo::InputParam*>(current_param);
                                if (input_param)
                                {
                                    if (current_node->connectInput(output_node, output_param, input_param, flags))
                                    {
                                        std::cout << "Successfully set input of " << current_param->getName() << " to "
                                                  << output_param->getName() << "\n";
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
                if (!current_param) {}
                else
                {
                    auto func = mo::SerializationFactory::instance()->getTextDeSerializationFunction(
                        current_param->getTypeInfo());
                    if (func)
                    {
                        std::stringstream ss;
                        ss << value;
                        mo::Mutex_t::scoped_lock lock(current_param->mtx());
                        if (func(current_param, ss))
                            std::cout << "Successfully set " << current_param->getTreeName() << " to " << value
                                      << std::endl;
                        else
                            std::cout << "Failed to set " << current_param->getTreeName() << " to " << value
                                      << std::endl;
                        return;
                    }
                    else
                    {
                        std::cout << "No text deserialization function found for "
                                  << current_param->getTypeInfo().name() << std::endl;
                    }
                }
                if (current_node)
                {
                    auto params = current_node->getParams();
                    for (auto& param : params)
                    {
                        auto pos = value.find(param->getName());
                        if (pos != std::string::npos && value.size() > param->getName().size() + 1 &&
                            value[param->getName().size()] == ' ')
                        {
                            // std::cout << "Setting value for parameter " << param->getName() << " to " <<
                            // value.substr(pos + param->getName().size() + 1) << std::endl;
                            // std::stringstream ss;
                            // ss << value.substr(pos + param->getName().size() + 1);
                            // Parameters::Persistence::Text::DeSerialize(&ss, param);
                            return;
                        }
                    }
                    std::cout << "Unable to find parameter by name for set string: " << value << std::endl;
                    auto pos = value.find("sync");
                    if (pos == 0)
                    {
                        pos = value.find(' ');
                        current_node->setSyncInput(value.substr(pos + 1));
                    }
                }
                else if (current_stream)
                {
                }
                std::cout << "Unable to set value to " << value << std::endl;
            },
            std::placeholders::_1));

        connections.push_back(manager.connect(slot, "set"));
        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&current_node](std::string name) {
                if (current_node)
                {
                    current_node->setTreeName(name);
                }
            },
            std::placeholders::_1));
        connections.push_back(manager.connect(slot, "rename"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&current_node, &current_stream](std::string name) {
                mo::RelayManager* mgr = nullptr;
                if (current_node)
                {
                    mgr = current_node->getGraph()->getRelayManager();
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
                    if (global_signal_manager)
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
                    if (idx >= 0 && static_cast<size_t>(idx) < relays.size())
                        relay = relays[static_cast<size_t>(idx)].get();
                }
                else if (relays.size() == 1)
                {
                    relay = relays[0].get();
                }

                mo::TSignalRelay<void(void)>* typed = dynamic_cast<mo::TSignalRelay<void(void)>*>(relay);
                if (typed)
                {
                    (*typed)();
                    return;
                }
                THROW(debug) << "Signal serialization needs to be reimplemented";
            },
            std::placeholders::_1));
        connections.push_back(manager.connect(slot, "emit"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [](std::string level) {
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
            },
            std::placeholders::_1));
        connections.push_back(manager.connect(slot, "log"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [](std::string directory) {
                int idx = 0;
                if (auto pos = directory.find(',') != std::string::npos)
                {
                    idx = boost::lexical_cast<int>(directory.substr(0, pos));
                    directory = directory.substr(pos + 1);
                }
                mo::MetaObjectFactory::instance().getObjectSystem()->AddLibraryDir(directory.c_str(),
                                                                                   static_cast<unsigned short>(idx));
            },
            std::placeholders::_1));

        connections.push_back(manager.connect(slot, "link"));

        slot = new mo::TSlot<void(std::string)>(std::bind(
            [](std::string ms) {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(boost::lexical_cast<int>(ms)));
            },
            std::placeholders::_1));

        connections.push_back(manager.connect(slot, "wait"));

        std::vector<std::string> command_list;
        slot = new mo::TSlot<void(std::string)>(std::bind(
            [&command_list](std::string filename) {
                std::ifstream ifs(filename);
                if (ifs.is_open())
                {
                    std::string line;
                    while (std::getline(ifs, line))
                    {
                        if (line[line.size() - 1] == '\n' || line[line.size() - 1] == '\r')
                            line = line.substr(0, line.size() - 1);
                        command_list.push_back(line);
                    }
                    if (command_list.size())
                        std::reverse(command_list.begin(), command_list.end());
                }
                else
                {
                    MO_LOG(warning) << "Unable to load scripting file: " << filename;
                }
            },
            std::placeholders::_1));

        connections.push_back(manager.connect(slot, "run"));
        if (vm.count("config"))
        {
            MO_LOG(info) << "Loading " << vm["config"].as<std::string>();
            if (vm.count("disable-input") != 0)
            {
                auto relay = manager.getRelay<void(std::string)>("load");
                (*relay)(vm["config"].as<std::string>() + " " + vm["preset"].as<std::string>());
            }
            else
            {
                std::stringstream ss;
                ss << "load " << vm["config"].as<std::string>() << " " << vm["preset"].as<std::string>();
                command_list.emplace_back(ss.str());
            }
        }
        if (vm.count("launch"))
        {
            std::stringstream ss;
            ss << "load " << vm["launch"].as<std::string>();
            command_list.emplace_back(ss.str());
        }

        if (vm.count("file"))
        {
            auto relay = manager.getRelay<void(std::string)>("load_file");
            if (relay)
            {
                std::string file = vm["file"].as<std::string>();
                (*relay)(file);
            }
        }

        print_options();
        bool compiling = false;
        bool rcc_enabled = !vm["disable-rcc"].as<bool>() && (vm.count("profile-for") == 0);
        if (rcc_enabled)
            mo::MetaObjectFactory::instance().checkCompile();
        auto compile_check_function = [&_Graphs, &compiling, rcc_enabled]() {
            if (rcc_enabled)
            {
                if (mo::MetaObjectFactory::instance().checkCompile())
                {
                    std::cout << "Recompiling...\n";
                    for (auto& ds : _Graphs)
                    {
                        ds->stopThread();
                    }
                    compiling = true;
                }
                if (compiling)
                {
                    if (!mo::MetaObjectFactory::instance().isCompileComplete())
                    {
                        std::cout << "Still compiling\n";
                    }
                    else
                    {
                        for (auto& ds : _Graphs)
                        {
                            ds->stopThread();
                        }
                        if (mo::MetaObjectFactory::instance().swapObjects())
                        {
                            std::cout << "Object swap success\n";
                            for (auto& ds : _Graphs)
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

        slot = new mo::TSlot<void(std::string)>(
            std::bind([&rcc_enabled](std::string value) { rcc_enabled = boost::lexical_cast<bool>(value); },
                      std::placeholders::_1));
        connections.push_back(manager.connect(slot, "rcc"));

        bool disable_input = vm["disable-input"].as<bool>() || vm.count("profile-for");

        if (vm.count("script"))
        {
            auto relay = manager.getRelay<void(std::string)>("run");
            if (relay)
            {
                std::string file = vm["script"].as<std::string>();
                (*relay)(file);
            }
        }
        int run_time = -1;
        if (vm.count("profile-for") != 0)
        {
            run_time = vm["profile-for"].as<int>();
        }
        boost::thread io_thread =
            boost::thread(std::bind([&_Graphs, &manager, run_time, &command_list, &disable_input, &print_options]() {
                auto start = boost::posix_time::microsec_clock::universal_time();
                while (!quit)
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
                        if (!disable_input)
                            if (std::cin.peek())
                                std::getline(std::cin, command_line);
                    }
                    if (!skip)
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
                                MO_LOG(debug) << "Running command (" << command << ") with arguments: " << rest;
                                (*relay)(rest);
                            }
                            catch (std::exception& e)
                            {
                                MO_LOG(warning) << "Executing command (" << command << ") with arguments: " << rest
                                                << " failed due to: "
                                                << "[" << typeid(e).name() << "] - " << e.what();
                            }
                            catch (...)
                            {
                                MO_LOG(warning) << "Executing command (" << command << ") with arguments: " << rest
                                                << " failed miserably";
                            }
                        }
                        else
                        {
                            if (command_line.size())
                            {
                                MO_LOG(warning) << "Invalid command: " << command_line;
                                print_options();
                            }
                        }
                    }
                    mo::ThreadSpecificQueue::run();
                    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
                    if (run_time != -1)
                    {
                        auto now = boost::posix_time::microsec_clock::universal_time();
                        if (boost::posix_time::time_duration(now - start).total_seconds() > run_time)
                        {
                            quit = true;
                        }
                    }
                }

                std::cout << "IO thread shutting down\n";
            }));
        mo::setThreadName(io_thread, "io-thread");

        boost::posix_time::ptime last_compile_check_time = boost::posix_time::microsec_clock::universal_time();

        signal(SIGINT, sig_handler);
        mo::setThisThreadName("RCC");
        while (!quit)
        {
            auto current_time = boost::posix_time::microsec_clock::universal_time();
            if (boost::posix_time::time_duration(current_time - last_compile_check_time).total_milliseconds() > 1000)
            {
                last_compile_check_time = current_time;
                compile_check_function();
            }
            else
            {
                boost::this_thread::sleep_for(boost::chrono::seconds(1));
            }
        }
        io_thread.~thread();
        gui_thread.interrupt();
        gui_thread.join();
        mo::ThreadSpecificQueue::cleanup();
        _Graphs.clear();
        MO_LOG(info, "Gui thread shut down complete");
        mo::ThreadPool::instance()->cleanup();
        MO_LOG(info, "Thread pool cleanup complete");
        MO_LOG(info, "Cleaning up singletons");
        std::cout << "Program exiting" << std::endl;
        return 0;
    }
    gui_thread.interrupt();
    gui_thread.join();
    mo::ThreadSpecificQueue::cleanup();
    MO_LOG(info, "Gui thread shut down complete, cleaning up thread pool");
    mo::ThreadPool::instance()->cleanup();
    MO_LOG(info, "Thread pool cleanup complete");
    std::cout << "Program exiting" << std::endl;
    return 0;
}
