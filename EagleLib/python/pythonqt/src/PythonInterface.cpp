#include "PythonInterface.h"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/NodeManager.h"
#include "EagleLib/Plugins.h"
#include "EagleLib/rcc/ObjectManager.h"
#include "EagleLib/frame_grabber_base.h"
#include "EagleLib/DataStreamManager.h"

#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/exceptions.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <PythonQt/PythonQt.h>

using namespace EagleLib;
using namespace EagleLib::python;
using namespace EagleLib::python::wrappers;



void EagleLib::python::wrappers::RegisterMetaTypes()
{
	qRegisterMetaType<NodePtr>("NodePtr");
	PythonQt::self()->registerCPPClass("NodePtr", "", "EagleLib", PythonQtCreateObject<NodeWrapper>);

	qRegisterMetaType<DataStream*>("DataStream");
	PythonQt::self()->registerCPPClass("DataStream", "", "EagleLib", PythonQtCreateObject<DataStreamWrapper>);
}
EaglePython::EaglePython()
{
	boost::filesystem::path currentDir = boost::filesystem::current_path();
#ifdef _MSC_VER
#ifdef _DEBUG
	currentDir = boost::filesystem::path(currentDir.string() + "/../../Debug");
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
}

void EaglePython::LoadPlugin(QString path)
{
	std::string path_ = path.toStdString();
	if(boost::filesystem::is_regular_file(path_))
	{
		EagleLib::loadPlugin(path_);
	}
}
QStringList EaglePython::ListPlugins()
{
	QStringList output;
	auto plugins = EagleLib::ListLoadedPlugins();
	for(auto& plugin : plugins)
	{
		output.append(QString::fromStdString(plugin));
	}
	return output;
}

QStringList EaglePython::ListDevices()
{
	QStringList output;
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
					for(auto& device : devices)
					{
						output.append(QString::fromStdString(device));
					}
				}
			}
		}				
	}
	return output;
}
QStringList EaglePython::ListConstructableNodes()
{
	return ListConstructableNodes("");
}
QStringList EaglePython::ListConstructableNodes(QString filter_)
{
	QStringList output;
	std::string filter = filter_.toStdString();
	auto nodes = EagleLib::NodeManager::getInstance().getConstructableNodes();
	for(auto& node : nodes)
	{
		if(filter.size())
		{
			if(node.find(filter) != std::string::npos)
			{
				output.append(QString::fromStdString(node));
			}
		}else
		{
			output.append(QString::fromStdString(node));
		}
	}
	return output;
}
void EaglePython::SetLogLevel(QString level)
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
}
bool EaglePython::CheckRecompile()
{
	return false;
}
void EaglePython::AbortRecompile()
{

}
DataStream* EaglePython::GetDataStream(int index)
{
	CV_Assert(index >= 0 && index < _streams.size());
	return _streams[index].get();
}

void EaglePython::ReleaseDataStream(int index)
{

}

QStringList EaglePython::ListDataStreams()
{
	QStringList output;
	for(auto& stream : _streams)
	{
		output.append(QString::fromStdString(stream->GetFrameGrabber()->GetSourceFilename()));
	}
	return output;
}

DataStream* EaglePython::OpenDataSource(QString source)
{
	std::string doc = source.toStdString();
	if(EagleLib::DataStream::CanLoadDocument(doc))
    {
        LOG(debug) << "Found a frame grabber which can load " << doc;
        auto stream = EagleLib::DataStreamManager::instance()->create_stream();
        if(stream->LoadDocument(doc))
        {
			stream->LaunchProcess();
            _streams.push_back(stream);
			return stream.get();
        }else
        {
            LOG(warning) << "Unable to load document";
        }
    }else
    {
        LOG(warning) << "Unable to find a frame grabber which can load " << doc;
    }
	return nullptr;
}
void NodeWrapper::delete_NodePtr(NodePtr* o) 
{ 
	delete o;
}
QString NodeWrapper::GetName(NodePtr* node)
{
	return QString::fromStdString((*node)->getName());
}

QString NodeWrapper::GetFullName(NodePtr* node)
{
	return QString::fromStdString((*node)->getFullTreeName());
}

void NodeWrapper::AddNode(rcc::shared_ptr<EagleLib::Nodes::Node>* node, QString name)
{
	auto new_node = EagleLib::NodeManager::getInstance().addNode(name.toStdString());
	if(new_node)
	{
		(*node)->addChild(new_node);
	}
	
}

QStringList NodeWrapper::ListParameters(rcc::shared_ptr<EagleLib::Nodes::Node>* node)
{
	auto params = (*node)->getDisplayParameters();
	QStringList output;
	for(auto& param : params)
	{
		output.append(QString::fromStdString(param->GetName()));
	}
	return output;
}

void DataStreamWrapper::Step(DataStream* stream)
{

}

void DataStreamWrapper::Play(DataStream* stream)
{

}

void DataStreamWrapper::Pause(DataStream* stream)
{

}

void DataStreamWrapper::AddNode(DataStream* stream, QString name)
{
	auto node = EagleLib::NodeManager::getInstance().addNode(name.toStdString());
	if(node)
	{
		stream->AddNode(node);
	}	
}

QStringList DataStreamWrapper::ListNodes(DataStream* stream)
{
	QStringList output;
	auto nodes = stream->GetNodes();	
	for(auto& node : nodes)
	{
		output.append(QString::fromStdString(node->getName()));
	}
	return output;
}

rcc::shared_ptr<EagleLib::Nodes::Node> DataStreamWrapper::GetNode(DataStream* stream, QString name)
{
	return NodePtr();
}

rcc::shared_ptr<EagleLib::Nodes::Node> DataStreamWrapper::GetNode(DataStream* stream, int index)
{
	auto nodes = stream->GetNodes();
	CV_Assert(index >= 0 && index < nodes.size());
	return rcc::shared_ptr<EagleLib::Nodes::Node>(nodes[index]);
}