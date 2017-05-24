#include "RosInterface.hpp"
#include "ros/init.h"
#include "ros/node_handle.h"

#include "Aquila/rcc/SystemTable.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

using namespace aq;

RosInterface::RosInterface()
{
    int argc = 1;
    char** argv = new char*{ "EagleEye"};
    ros::init(argc, argv, "EagleEye");
    _nh = new ros::NodeHandle("EagleEye");
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
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    RosInterface* singleton = table->getSingleton<RosInterface>();
    if(singleton)
    {
        return singleton;
    }else
    {
        singleton = new RosInterface();
        table->SetSingleton(singleton);
        return singleton;
    }
}
