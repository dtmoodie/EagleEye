#include "Stereo.h"
#include "external_includes/cv_imgproc.hpp"
#include "external_includes/cv_cudawarping.hpp"

#if _WIN32
    #if _DEBUG
        RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300d.lib")
    #else
        RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300.lib")
    #endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudastereo")
#endif



using namespace EagleLib;

void StereoBM::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Num disparities", int(64));
        updateParameter("Block size", int(19));
        addInputParameter<cv::cuda::GpuMat>("Left image");
        addInputParameter<cv::cuda::GpuMat>("Right image");
    }
    stereoBM = cv::cuda::createStereoBM(*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data());
}

cv::cuda::GpuMat StereoBM::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed || parameters[1]->changed)
    {
        stereoBM = cv::cuda::createStereoBM(*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data());
    }
    cv::cuda::GpuMat* left = getParameter<cv::cuda::GpuMat>(2)->Data();
    cv::cuda::GpuMat* right = getParameter<cv::cuda::GpuMat>(3)->Data();
    if(left == nullptr)
    {
        left = &img;
    }
    if(right == nullptr)
    {
        //log(Error, "No input selected for right image");
		NODE_LOG(error) << "No input selected for right image";
        return img;
    }
    if(left->size() != right->size())
    {
        //log(Error, "Images are of mismatched size");
		NODE_LOG(error) << "Images are of mismatched size";
        return img;
    }
    if(left->channels() != right->channels())
    {
        //log(Error, "Images are of mismatched channels");
		NODE_LOG(error) << "Images are of mismatched channels";
        return img;
    }
    auto buf = disparityBuf.getFront();

    stereoBM->compute(*left,*right,buf->data, stream);
    buf->record(stream);
    return buf->data;
}

void StereoBilateralFilter::Init(bool firstInit)
{

}

cv::cuda::GpuMat StereoBilateralFilter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}

void StereoBeliefPropagation::Init(bool firstInit)
{
    bp = cv::cuda::createStereoBeliefPropagation();
}

cv::cuda::GpuMat StereoBeliefPropagation::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}

void StereoConstantSpaceBP::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter<int>("Num disparities", 128);
        updateParameter<int>("Num iterations", 8);
        updateParameter<int>("Num levels", 4);
        updateParameter<int>("NR plane", 4);
        Parameters::EnumParameter param;
        param.addEnum(ENUM(CV_16SC1));
        param.addEnum(ENUM(CV_32FC1));
        updateParameter("Message type", param);
        //createStereoConstantSpaceBP(int ndisp = 128, int iters = 8, int levels = 4, int nr_plane = 4, int msg_type = CV_32F);
        addInputParameter<cv::cuda::GpuMat>("Left image");
        addInputParameter<cv::cuda::GpuMat>("Right image");
        csbp = cv::cuda::createStereoConstantSpaceBP();
    }else
    {
        parameters[0]->changed = true;
    }

}

cv::cuda::GpuMat StereoConstantSpaceBP::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed || parameters[1]->changed || parameters[2]->changed || parameters[3]->changed || parameters[4]->changed)
    {
        csbp = cv::cuda::createStereoConstantSpaceBP(*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data(), *getParameter<int>(2)->Data(), *getParameter<int>(3)->Data(), getParameter<Parameters::EnumParameter>(4)->Data()->getValue());
    }
    if(csbp == nullptr)
    {
        //log(Error, "Stereo constant space bp == nullptr");
		NODE_LOG(error) << "Stereo constant space bp == nullptr";
        return img;
    }
    cv::cuda::GpuMat* left, *right;
    left = getParameter<cv::cuda::GpuMat>("Left image")->Data();
    right = getParameter<cv::cuda::GpuMat>("Right image")->Data();
    if(left == nullptr)
        left = & img;
    if(right == nullptr)
    {
        //log(Error, "Right image input not defined");
		NODE_LOG(error) << "Right image input not defined";
        return img;
    }
    cv::cuda::GpuMat disp;
    csbp->compute(*left, *right,disp);
    return disp;
}
void UndistortStereo::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::Mat>("Camera Matrix");
        addInputParameter<cv::Mat>("Distortion Matrix");
        addInputParameter<cv::Mat>("Rotation Matrix");
        addInputParameter<cv::Mat>("Projection Matrix");
        updateParameter<cv::cuda::GpuMat>("mapX", cv::cuda::GpuMat());
        updateParameter<cv::cuda::GpuMat>("mapY", cv::cuda::GpuMat());
    }
}

cv::cuda::GpuMat UndistortStereo::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed || parameters[1]->changed || parameters[2]->changed || parameters[3]->changed)
    {
        cv::Mat* K = getParameter<cv::Mat>(0)->Data();
        if(K == nullptr)
        {
            //log(Warning, "Camera matrix undefined");
			NODE_LOG(warning) << "Camera matrix undefined";
            return img;
        }
        cv::Mat* D = getParameter<cv::Mat>(1)->Data();
        if(D == nullptr)
        {
            //log(Warning, "Distortion matrix undefined");
			NODE_LOG(warning) << "Distortion matrix undefined";
            return img;
        }
        cv::Mat* R = getParameter<cv::Mat>(2)->Data();
        if(R == nullptr)
        {
            //log(Warning, "Rotation matrix undefined");
			NODE_LOG(warning) << "Rotation matrix undefined";
            return img;
        }
        cv::Mat* P = getParameter<cv::Mat>(3)->Data();
        if(P == nullptr)
        {
            //log(Warning, "Projection matrix undefined");
			NODE_LOG(warning) << "Projection matrix undefined";
            return img;
        }
        if(K->empty())
        {
            //log(Warning, "Camera matrix empty");
			NODE_LOG(warning) << "Camera matrix empty";
            return img;
        }
        if(D->empty())
        {
            //log(Warning, "Distortion matrix empty");
			NODE_LOG(warning) << "Distortion matrix empty";
            return img;
        }
        if(R->empty())
        {
            //log(Warning, "Rotation matrix empty");
			NODE_LOG(warning) << "Rotation matrix empty";
            return img;
        }
        if(P->empty())
        {
            //log(Warning, "Projection matrix empty");
			NODE_LOG(warning) << "Projection matrix empty";
            return img;
        }

        //log(Status, "Calculating image rectification");
		NODE_LOG(info) << "Calculating image rectification";
        cv::initUndistortRectifyMap(*K,*D, *R, *P, img.size(), CV_32FC1, X, Y);
        mapX.upload(X, stream);
        mapY.upload(Y,stream);
        //log(Status, "Undistortion maps calculated");
		NODE_LOG(info) << "Undistortion maps calculated";
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
        updateParameter("mapX", mapX);
        updateParameter("mapY", mapY);

    }
    if(!mapX.empty() && !mapY.empty())
    {
        cv::cuda::remap(img,img,mapX,mapY, CV_INTER_CUBIC, cv::BORDER_REPLICATE, cv::Scalar(), stream);
    }
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBM)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBilateralFilter)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBeliefPropagation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoConstantSpaceBP)
NODE_DEFAULT_CONSTRUCTOR_IMPL(UndistortStereo)
