/*  Serial stack nodes execute their children in the order that they are listed
 *
 *
 *
 *
 *
 *
*/
#include "Node.h"
namespace EagleLib
{
    class CV_EXPORTS SerialStack: public Node
    {
    public:
        SerialStack();
        ~SerialStack();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);

    };


}
