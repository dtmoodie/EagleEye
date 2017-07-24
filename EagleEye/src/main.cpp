#include "mainwindow.h"
#include <QApplication>
#include "Aquila/rcc/SystemTable.hpp"
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/core.hpp>
#include <boost/log/exceptions.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <MetaObject/MetaParameters.hpp>
void loadDir(boost::filesystem::path path){
    boost::filesystem::directory_iterator end_itr;
    if(boost::filesystem::is_directory(path)){
        for(boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr){
            if(boost::filesystem::is_regular_file(itr->path())){
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
}

int main(int argc, char *argv[])
{
    mo::MetaParams::initialize();
    SystemTable table;
    mo::MetaObjectFactory::instance(&table);
#if QT_VERSION > 0x050400
    QApplication::setAttribute(Qt::AA_ShareOpenGLContexts, true);
#endif
    QApplication a(argc, argv);

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("log", boost::program_options::value<std::string>()->default_value("info"), "Logging verbosity. trace, debug, info, warning, error, fatal")
        ("plugins", boost::program_options::value<boost::filesystem::path>(), "Path to additional plugins to load")
        ("file", boost::program_options::value<std::string>(), "Path to file to initialize with")
        ("preferred_loader", boost::program_options::value<std::string>(), "Preferred loader to initialize with")
        ("log", boost::program_options::value<std::string>()->default_value("info"), "Logging verbosity. trace, debug, info, warning, error, fatal");

    boost::program_options::variables_map vm;
    if (vm.count("log")) {
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
    else {
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
    }
    auto parsed_options = boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    boost::program_options::store(parsed_options, vm);
    
    auto unrecognized = boost::program_options::collect_unrecognized(parsed_options.options, boost::program_options::include_positional);
    std::map<std::string, std::string> replace_map;
    std::map<std::string, std::string> variable_replace_map;
    for (auto& option : unrecognized) {
        auto pos = option.find(":=");
        if (pos != std::string::npos) {
            std::string start = option.substr(0, pos);
            std::string end = option.substr(pos + 2);
            if (end[0] == '\"' && end[end.size() - 1] == '\"') {
                end = end.substr(1, end.size() - 2);
            }
            replace_map["${" + start + "}"] = end;
            continue;
        }
        pos = option.find("=");
        if (option.find("--") == 0 && pos != std::string::npos) {
            std::string var_name = option.substr(2, pos - 2);
            std::string var_value = option.substr(pos + 1);
            if (var_value[0] == '\"' && var_value[var_value.size() - 1] == '\"') {
                var_value = var_value.substr(1, var_value.size() - 2);
            }
            variable_replace_map[var_name] = var_value;
        }
    }
    if (replace_map.size()) {
        std::stringstream ss;
        for (const auto& pair : replace_map)
            ss << "\n"
            << pair.first << " = " << pair.second;
        MO_LOG(debug) << "Input string replacements: " << ss.str();
    }

    if (variable_replace_map.size()) {
        std::stringstream ss;
        for (const auto& pair : variable_replace_map)
            ss << "\n"
            << pair.first << " = " << pair.second;
        MO_LOG(debug) << "Input variable replacements: " << ss.str();
    }

    boost::filesystem::path current_dir("./"); // = boost::filesystem::path(argv[0]).parent_path();
    current_dir = boost::filesystem::path(current_dir.string() + "/Plugins");
    MO_LOG(info) << "Looking for plugins in: " << current_dir.string();
    loadDir(current_dir);
    current_dir = boost::filesystem::path(argv[0]).parent_path();
    current_dir = boost::filesystem::path(current_dir.string() + "/Plugins");
    MO_LOG(info) << "Looking for plugins in: " << current_dir.string();
    loadDir(current_dir);
    if(vm.count("plugins"))
    {
        auto currentDir = boost::filesystem::path(vm["plugins"].as<boost::filesystem::path>());
        boost::filesystem::directory_iterator end_itr;
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
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    rcc::shared_ptr<MainWindow> w = rcc::shared_ptr<MainWindow>::create();
    w->setVmSm(&variable_replace_map, &replace_map);
    w->show();

    return a.exec();
}

