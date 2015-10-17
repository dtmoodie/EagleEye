#include "KeyFrameServer.h"

using namespace EagleLib;


template<typename T> bool KeyFrameServer::getParameter(int frameIndex, KeyFrameParameter param, T* data)
{
    auto itr = KeyFrameParameters.find(frameIndex);
    if(itr == KeyFrameParameters.end())
        return false;

    auto itr2 = itr->second.find(param);
    if(itr2 == itr->second.end())
        return false;

    // Now attempt to cast to the appropriate type
	typename Parameters::ITypedParameter<T>::Ptr paramPtr = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(itr2->second);
    if(paramPtr == nullptr)
        return false;

    data = paramPtr->Data();
    return true;
}
#define logParam( param ) case param: logMsg << #param; break;
template<typename T> bool KeyFrameServer::setParameter(int frameIndex, KeyFrameParameter param, const T& data)
{
	std::map<KeyFrameParameter, Parameters::Parameter::Ptr>& keyFrame = KeyFrameParameters[frameIndex];
	keyFrame[param] = typename Parameters::ITypedParameter<T>::Ptr(new Parameters::TypedParameter<T>("", data));
    std::stringstream logMsg;
    switch(param)
    {
        logParam(Image)
        logParam(CameraMatrix)
        logParam(GreyScale)
        logParam(KeyPoints2D)
        logParam(TrackedPoints2D)
        logParam(TrackedPointMask)
        logParam(Pose)
        logParam(CorrespondingChildFrames)
        logParam(CorrespondingKeyFrames)
            default:
            logMsg << "Parameter idx: " << param;
    }
    logMsg << " updated for keyframe " << frameIndex;
    NODE_LOG(info) << logMsg.str();
    return true;
}


void KeyFrameServer::Init(bool firstInit)
{
    /*updateParameter<d_ParameterSetFunctor>("Update Device Parameter",       boost::bind(&KeyFrameServer::setParameter<cv::cuda::GpuMat>,this,_1,_2,_3));
    updateParameter<h_ParameterSetFunctor>("Update Host Parameter",         boost::bind(&KeyFrameServer::setParameter<cv::Mat>,this,_1,_2,_3));
    updateParameter<h_KeyPointSetFunctor>("Update KeyPoint Parameter",      boost::bind(&KeyFrameServer::setParameter<std::vector<cv::KeyPoint>>,this,_1,_2,_3));

    updateParameter<d_ParameterGetFunctor>("Get Device Parameter",          boost::bind(&KeyFrameServer::getParameter<cv::cuda::GpuMat>,this,_1,_2,_3));
    updateParameter<h_ParameterGetFunctor>("Get Host Parameter",            boost::bind(&KeyFrameServer::getParameter<cv::Mat>,this,_1,_2,_3));
    updateParameter<h_KeyPointGetFunctor>("Get Keypoint Parameter",         boost::bind(&KeyFrameServer::getParameter<std::vector<cv::KeyPoint>>,this,_1,_2,_3));*/

}

cv::cuda::GpuMat KeyFrameServer::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    updateParameter("Num keyframes tracked", KeyFrameParameters.size());
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(KeyFrameServer)
