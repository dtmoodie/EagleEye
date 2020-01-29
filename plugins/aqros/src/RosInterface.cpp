#include "RosInterface.hpp"
#include "ros/init.h"
#include "ros/node_handle.h"

#include <MetaObject/core/SystemTable.hpp>

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

#include <MetaObject/core/metaobject_config.hpp>

#include <Aquila/core/KeyValueStore-inl.hpp>

using namespace aq;

RosInterface::RosInterface()
{
    std::string node_name = "EagleEye";
    node_name = aq::KeyValueStore::instance().read("node_name", node_name);
    MO_LOG(info) << "Creating ros nodr handle with node name:" << node_name;
    int argc = 1;
    std::vector<char*> names;
    names.push_back(const_cast<char*>(node_name.c_str()));
    ros::init(argc, names.data(), node_name);
    _nh = new ros::NodeHandle(node_name);
}

RosInterface::~RosInterface()
{
    delete _nh;
}

ros::NodeHandle* RosInterface::nh() const
{
    return _nh;
}

RosInterface* RosInterface::Instance()
{
    return singleton<RosInterface>();
}

extern "C" {
void InitModule()
{
}
}
