#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace aq
{
    namespace nodes
    {
        // Initial implementaiton of birds eye view transform from
        // https://arxiv.org/pdf/1611.07759.pdf
        class BirdsEye : public Node
        {
          public:
            MO_DERIVE(BirdsEye, Node)
            INPUT(pcl::PointCloud<pcl::PointXYZI>::Ptr, point_cloud, nullptr)
            PARAM(float, resolution, 0.1)
            PARAM(int, width, 1024)
            PARAM(int, height, 1024)
            PARAM(int, slices, 1)
            PARAM(float, min_z, 0)
            PARAM(float, max_z, 10)
            OUTPUT(SyncedMemory, birds_eye_view, {})
            MO_END;

          protected:
            bool processImpl();
        };
    }
}
