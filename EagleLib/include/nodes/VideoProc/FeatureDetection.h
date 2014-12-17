#include <nodes/Node.h>

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace EagleLib
{

    class GoodFeaturesToTrackDetector: public Node
    {
    public:
        GoodFeaturesToTrackDetector();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);

        boost::shared_ptr< TypedParameter< cv::Ptr<cv::cuda::CornersDetector> > > detector;
        boost::shared_ptr< InputParameter< int > > numCorners;
        boost::shared_ptr< InputParameter< double > > qualityLevel;
        boost::shared_ptr< InputParameter< double > > minDistance;
        boost::shared_ptr< InputParameter< int > > blockSize;
        boost::shared_ptr< InputParameter< bool > > useHarris;
        boost::shared_ptr< InputParameter< double> > harrisK;
        boost::shared_ptr< InputParameter< bool > > calculateFlag;
        boost::shared_ptr< OutputParameter< cv::cuda::GpuMat> >corners;
        int imgType;

    };
}
