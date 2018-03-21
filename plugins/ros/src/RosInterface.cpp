#include "RosInterface.hpp"
#include "ros/init.h"
#include "ros/node_handle.h"

#include <MetaObject/core/SystemTable.hpp>

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

using namespace aq;

RosInterface::RosInterface()
{
    int argc = 1;
    char** argv = new char* {"EagleEye"};
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
    auto singleton = table->getSingleton<RosInterface>();
    if (singleton)
    {
        return singleton.get();
    }
    else
    {
        singleton.reset(new RosInterface());
        table->setSingleton(singleton);
        return singleton.get();
    }
}
