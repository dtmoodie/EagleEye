#include "Stereo.h"
#include "EagleLib/rcc/external_includes/cv_imgproc.hpp"
#include "EagleLib/rcc/external_includes/cv_cudawarping.hpp"

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
using namespace EagleLib::Nodes;


bool StereoBM::ProcessImpl()
{
    if(!stereoBM || num_disparities_param.modified || block_size_param.modified)
    {
        stereoBM = cv::cuda::createStereoBM(num_disparities, block_size);
        block_size_param.modified = false;
        num_disparities_param.modified = false;
    }
    if (left_image->GetSize() != right_image->GetSize())
    {
        //log(Error, "Images are of mismatched size");
        LOG(debug) << "Images are of mismatched size";
        return false;
    }
    cv::cuda::GpuMat disparity;
    stereoBM->compute(left_image->GetGpuMat(*_ctx->stream), right_image->GetGpuMat(*_ctx->stream),disparity, *_ctx->stream);
    this->disparity_param.UpdateData(disparity, left_image_param.GetTimestamp(), _ctx);
}

void StereoBeliefPropagation::NodeInit(bool firstInit)
{
    bp = cv::cuda::createStereoBeliefPropagation();
}

cv::cuda::GpuMat StereoBeliefPropagation::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}

void StereoConstantSpaceBP::NodeInit(bool firstInit)
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
        _parameters[0]->changed = true;
    }

}
bool StereoConstantSpaceBP::ProcessImpl()
{
    if(num_levels_param.modified || nr_plane_param.modified || 
        num_disparities_param.modified || num_iterations_param.modified || !csbp)
    {
        csbp = cv::cuda::createStereoConstantSpaceBP(
            num_disparities, 
            num_iterations, 
            num_levels, nr_plane, message_type.getValue());
        num_levels_param.modified = false;
        nr_plane_param.modified =  false;
        num_disparities_param.modified = false;
        num_iterations_param.modified = false;
    }
    cv::cuda::GpuMat disparity;
    csbp->compute(left_image->GetGpuMat(*_ctx->stream), right_image->GetGpuMat(*_ctx->stream),disparity, *_ctx->stream);
    this->disparity_param.UpdateData(disparity, left_image_param.GetTimestamp(), _ctx);
}

void UndistortStereo::NodeInit(bool firstInit)
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
bool UndistortStereo::ProcessImpl()
{
    if(camera_matrix_param.modified || distortion_matrix_param.modified ||
        rotation_matrix_param.modified || projection_matrix_param.modified)
    {
        cv::Mat X, Y;
        cv::initUndistortRectifyMap(*camera_matrix, *distortion_matrix,
            *rotation_matrix, *projection_matrix, input->GetSize(), CV_32FC1, X, Y);
        mapX_param.UpdateData(X, -1, _ctx);
        mapY_param.UpdateData(Y, -1, _ctx);
    }
    cv::cuda::remap(input->GetGpuMat(*_ctx->stream), input->GetGpuMatMutable(*_ctx->stream), 
        mapX.GetGpuMat(*_ctx->stream), mapY.GetGpuMat(*_ctx->stream), 
        CV_INTER_CUBIC, cv::BORDER_REPLICATE, cv::Scalar(), *_ctx->stream);
    return true;
}
cv::cuda::GpuMat UndistortStereo::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(_parameters[0]->changed || _parameters[1]->changed || _parameters[2]->changed || _parameters[3]->changed)
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
        _parameters[0]->changed = false;
        _parameters[1]->changed = false;
        _parameters[2]->changed = false;
        _parameters[3]->changed = false;
        updateParameter("mapX", mapX);
        updateParameter("mapY", mapY);

    }
    if(!mapX.empty() && !mapY.empty())
    {
        cv::cuda::remap(img,img,mapX,mapY, CV_INTER_CUBIC, cv::BORDER_REPLICATE, cv::Scalar(), stream);
    }
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBM, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBilateralFilter, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBeliefPropagation, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoConstantSpaceBP, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(UndistortStereo, Image, Processing)
