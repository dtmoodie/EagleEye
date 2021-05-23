#include <aqcore_export.hpp>

#include <Aquila/nodes/Node.hpp>

#include <Aquila/rcc/external_includes/cv_cudafeatures3d.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>

#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedImage.hpp>
#include <Aquila/types/geometry/Circle.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class Canny : public Node
        {
            cv::Ptr<cv::cuda::CannyEdgeDetector> detector;

          public:
            MO_DERIVE(Canny, Node)
                PARAM(double, low_thresh, 0.0)
                PARAM(double, high_thresh, 20.0)
                PARAM(int, aperature_size, 3)
                PARAM(bool, L2_gradient, false)

                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, edges)
            MO_END

          protected:
            virtual bool processImpl() override;
        };

    } // namespace nodes
} // namespace aq
