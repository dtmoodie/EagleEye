#include "interface_node.h"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/NodeManager.h"

using namespace EagleLib;
using namespace EagleLib::python;

python::NodeManager::NodeManager()
{

}

QStringList python::NodeManager::ListConstructableNodes()
{
	auto nodes = EagleLib::NodeManager::getInstance().getConstructableNodes();
	QStringList output;
	for(auto& node : nodes)
	{
		output.append(QString::fromStdString(node));
	}
	return output;
}

python::DataStream::DataStream()
{

}

QStringList python::DataStream::ListChildren()
{
	QStringList children;

	return children;
}

QStringList python::DataStream::ListParameters()
{
	QStringList parameters;


	return parameters;
}

python::Node::Node()
{

}

QStringList python::Node::ListChildren()
{
	QStringList children;


	return children;
}

QStringList python::Node::ListParameters()
{
	QStringList parameters;


	return parameters;
}