#pragma once


#include "ROSExport.hpp"

namespace ros
{
    class NodeHandle;
}
namespace EagleLib
{
    /*!
     * \brief The RosInterface class is the boundary between eagle eye and ros
     *        it allows for a unique node handle and prevents multiple initialization of
     *        ros
     */
    class ROS_EXPORT RosInterface
    {
    public:
        static RosInterface* Instance();
        ros::NodeHandle* nh() const;
        ~RosInterface();
    protected:
        RosInterface();

        ros::NodeHandle* _nh;
    };
}
