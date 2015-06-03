#pragma once

#include "nodes/Node.h"

namespace EagleLib
{
    class CorrespondenceServer: public Node
    {
    public:
        enum CorrespondenceType
        {

        };



    };

    class KeyFrameServer: public Node
    {

    public:
        enum KeyFrameParameter
        {
            Image = 0,
            CameraMatrix,
            GreyScale,
            KeyPoints2D,
            TrackedPoints2D,
            TrackedPointMask,
            Pose,
            CorrespondingChildFrames,  // cv::Mat of unsigned integers for each child frame that this keyframe has a correspondence to
            CorrespondingKeyFrames     // cv::Mat of unsigned integers for each other key frame that this keyframe has a correspondence to
        };
        //typedef boost::property_tree::basic_ptree<std::string, EagleLib::Parameter::Ptr> ParameterTree;

        //ParameterTree KeyFrameParameters;
        std::map<int, std::map<KeyFrameParameter, EagleLib::Parameter::Ptr> > KeyFrameParameters;

        typedef boost::function<bool(int, KeyFrameParameter, const cv::cuda::GpuMat&)>              d_ParameterSetFunctor;
        typedef boost::function<bool(int, KeyFrameParameter, const cv::Mat&)>                       h_ParameterSetFunctor;
        typedef boost::function<bool(int, KeyFrameParameter, const std::vector<cv::KeyPoint>&)>     h_KeyPointSetFunctor;

        typedef boost::function<bool(int, KeyFrameParameter, cv::cuda::GpuMat&)>              d_ParameterGetFunctor;
        typedef boost::function<bool(int, KeyFrameParameter, cv::Mat&)>                       h_ParameterGetFunctor;
        typedef boost::function<bool(int, KeyFrameParameter, std::vector<cv::KeyPoint>&)>     h_KeyPointGetFunctor;

        template<typename T> bool getParameter(int frameIndex, KeyFrameParameter param, T& data);
        template<typename T> bool setParameter(int frameIndex, KeyFrameParameter param, const T& data);



//        bool getParameter(int frameIndex, KeyFrameParameter param, cv::cuda::GpuMat& data);
//        bool getParameter(int frameIndex, KeyFrameParameter param, cv::Mat& data);
//        bool getParameter(int frameIndex, KeyFrameParameter param, std::vector<cv::KeyPoint>& data);

//        bool setParameter(int frameIndex, KeyFrameParameter param, const cv::cuda::GpuMat& data);
//        bool setParameter(int frameIndex, KeyFrameParameter param, const cv::Mat& data);
//        bool setParameter(int frameIndex, KeyFrameParameter param, const std::vector<cv::KeyPoint>& data);

        KeyFrameServer();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
