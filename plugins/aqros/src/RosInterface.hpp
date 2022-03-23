#pragma once
#include <MetaObject/core/SystemTable.hpp>

#include <memory>
namespace ros
{
    class NodeHandle;
}
namespace aq
{
    /*!
     * \brief The RosInterface class is the boundary between eagle eye and ros
     *        it allows for a unique node handle and prevents multiple initialization of
     *        ros
     */
    class RosInterface
    {
      public:
        static std::shared_ptr<RosInterface> Instance();
        ros::NodeHandle* nh() const;
        RosInterface();
        ~RosInterface();

      protected:
        ros::NodeHandle* _nh;
    };
} // namespace aq

extern "C" {
void initModule(SystemTable*);
}
