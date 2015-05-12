#include <boost/function.hpp>
<<<<<<< HEAD
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace EagleLib
{
typedef boost::function<int(cv::cuda::GpuMat,cv::cuda::GpuMat,cv::cuda::Stream,int)> setReferenceFunctor;
typedef boost::function<cv::cuda::GpuMat(cv::cuda::GpuMat, cv::cuda::GpuMat*, cv::cuda::GpuMat*)> trackFunctor;
}

=======
#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>

#include <opencv2/core/cuda.hpp>
#include <map>

namespace EagleLib
{
    typedef
boost::function<cv::cuda::GpuMat(
    cv::cuda::GpuMat,       // reference image
    cv::cuda::GpuMat,       // Current image
    cv::cuda::GpuMat,       // Reference points
    cv::cuda::GpuMat,       // Estimated current image points
    cv::cuda::GpuMat&,      // status output
    cv::cuda::GpuMat&,      // error output
    cv::cuda::Stream)>      // stream
        TrackSparseFunctor;

    class Correspondence
    {
        int frameIndex;
        int keyFrameIndex;

    };


    class KeyFrame
    {
        enum VariableType
        {
            Pose = 0,
            CameraMatrix,
            Homography,
            CoordinateSystem
        };
        std::map<VariableType, boost::any> data;
        std::map<int, boost::shared_ptr<Correspondence>> correspondences;
    public:
        KeyFrame(cv::cuda::GpuMat img_, int idx_);
        cv::cuda::GpuMat img;
        int frameIndex;

        bool setPose(cv::Mat pose);
        bool setPoseCoordinateSystem(std::string coordinateSyste);
        bool setCamera(cv::Mat cameraMatrix);
        bool setCorrespondene(int otherIdx, cv::cuda::GpuMat thisFramePts, cv::cuda::GpuMat otherFramePts);
        bool getCorrespondence(int otherIdx, cv::cuda::GpuMat& thisFramePts, cv::cuda::GpuMat& otherFramePts);
        bool getCorrespondence(int otherIdx, cv::Mat& homography);
        bool getHomography(int otherIdx, cv::Mat& homography);
        bool hasCorrespondence(int otherIdx);
    };
}
>>>>>>> 591e3beeebd8738622ec58f3a2913592780a1ecd
