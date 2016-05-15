
#include <EagleLib/rcc/ObjectManager.h>
#include <EagleLib/nodes/NodeManager.h>
#include <EagleLib/nodes/Node.h>
#include <EagleLib/frame_grabber_base.h>


#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>




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



BOOST_PYTHON_MODULE(EaglePython)
{
	boost::python::scope().attr("__version__") = "0.1";

	boost::python::def("ListConstructableNodes", ListConstructableNodes);
	boost::python::def("ListConstructableNodes", ListConstructableNodes1);

}