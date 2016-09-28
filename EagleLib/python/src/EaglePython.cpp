#include <EagleLib/nodes/NodeManager.h>
#include <EagleLib/nodes/Node.h>
#include <EagleLib/frame_grabber_base.h>
#include <EagleLib/DataStreamManager.h>

#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace std;
using namespace EagleLib;


// --------------------------------------------------------------
// Here are a bunch of static functions that can be called from python

std::vector<std::string> ListConstructableNodes(std::string filter)
{
    std::vector<std::string> output;
    auto nodes = EagleLib::NodeManager::getInstance().getConstructableNodes();
    for(auto& node : nodes)
    {
        if(filter.size())
        {
            if(node.find(filter) != std::string::npos)
            {
                output.push_back(node);
            }
        }else
        {
            output.push_back(node);
        }
    }
    return output;
}
std::vector<std::string> ListConstructableNodes1()
{
    return ListConstructableNodes("");
}
std::vector<std::string> ListDevices()
{
    std::vector<std::string> output;
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
                        output.push_back(device);
                    }
                }
            }
        }                
    }
    return output;
}

std::vector<std::string> ListHistory()
{
    return std::vector<std::string>();
}

rcc::shared_ptr<IDataStream> open_datastream(string source)
{
    std::string doc = source;
    if(EagleLib::IDataStream::CanLoadDocument(doc))
    {
        LOG(debug) << "Found a frame grabber which can load " << doc;
        auto stream = EagleLib::IDataStream::create(doc);
        if(stream->LoadDocument(doc))
        {
            stream->StartThread();
            return stream;
        }else
        {
            LOG(warning) << "Unable to load document";
        }
    }else
    {
        LOG(warning) << "Unable to find a frame grabber which can load " << doc;
    }
    return rcc::shared_ptr<IDataStream>();
}

rcc::shared_ptr<Nodes::Node> create_node(string name)
{
    return EagleLib::NodeManager::getInstance().addNode(name);
}

namespace boost
{
    namespace python
    {
        template<typename T> struct pointee<rcc::shared_ptr<T>>
        {
            typedef T type;
        };
    }
}
namespace rcc
{
    template<typename T> T* get_pointer(rcc::shared_ptr<T> const& p)
    {
        return p.get();
    }

}


BOOST_PYTHON_MODULE(EaglePython)
{
    boost::python::scope().attr("__version__") = "0.1";

    boost::python::def("ListConstructableNodes", &ListConstructableNodes);
    boost::python::def("ListConstructableNodes", &ListConstructableNodes1);
    boost::python::def("ListDevices", &ListDevices);


    //boost::python::class_<EagleLib::DataStream, rcc::shared_ptr<EagleLib::DataStream>, boost::noncopyable>("DataStream", boost::python::no_init)
        //.def("__init__", boost::python::make_constructor(&open_datastream))
        //.def("GetName", &EagleLib::Nodes::Node::getName)
        //.def("GetFullName", &EagleLib::Nodes::Node::getFullTreeName);
        //.def("GetParameters", &EagleLib::Nodes::Node::getParameters);

    boost::python::class_<EagleLib::Nodes::Node, rcc::shared_ptr<EagleLib::Nodes::Node>, boost::noncopyable>("Node", boost::python::no_init)
        .def("__init__", boost::python::make_constructor(&create_node));


    //boost::python::class_<Parameters::Parameter, boost::noncopyable>("Parameter", boost::python::no_init)
      //  .def("GetName", &Parameters::Parameter::GetName);
        
    
    //boost::python::register_ptr_to_python<rcc::shared_ptr<EagleLib::DataStream>>();

    boost::python::register_ptr_to_python<rcc::shared_ptr<EagleLib::Nodes::Node>>();

    boost::python::class_<vector<Parameters::Parameter*>>("ParamVec")
        .def(boost::python::vector_indexing_suite<vector<Parameters::Parameter*>, true>());

}
