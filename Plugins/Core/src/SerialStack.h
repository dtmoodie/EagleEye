/*  Serial stack nodes execute their children in the order that they are listed
 *
 *
 *
 *
 *
 *
*/
#include "Aquila/nodes/Node.hpp"
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace nodes
    {
    
    class CV_EXPORTS SerialStack: public Node
    {
    public:
        SerialStack();
        ~SerialStack();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual bool SkipEmpty() const;
    };
    }
}
