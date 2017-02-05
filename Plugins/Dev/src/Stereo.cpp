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
    stereoBM->compute(left_image->GetGpuMat(Stream()), right_image->GetGpuMat(Stream()),disparity, Stream());
    this->disparity_param.UpdateData(disparity, left_image_param.GetTimestamp(), _ctx);
    return true;
}

bool StereoBeliefPropagation::ProcessImpl()
{
    if(bp == nullptr ||
        num_iters_param.modified ||
        num_disparities_param.modified ||
        num_levels_param.modified,
        message_type_param.modified)
    {
        bp = cv::cuda::createStereoBeliefPropagation(num_disparities, num_iters, num_levels, message_type.getValue());

        num_iters_param.modified = true;
        num_disparities_param.modified = true;
        num_levels_param.modified = true;
        message_type_param.modified = true;
    }
    cv::cuda::GpuMat disparity;
    bp->compute(left_image->GetGpuMat(Stream()), right_image->GetGpuMat(Stream()), disparity, Stream());
    disparity_param.UpdateData(disparity, left_image_param.GetTimestamp(), _ctx);
    return true;
}



/*void StereoConstantSpaceBP::NodeInit(bool firstInit)
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
}*/
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
    csbp->compute(left_image->GetGpuMat(Stream()), right_image->GetGpuMat(Stream()),disparity, Stream());
    this->disparity_param.UpdateData(disparity, left_image_param.GetTimestamp(), _ctx);
    return true;
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
    cv::cuda::GpuMat remapped;
    cv::cuda::remap(input->GetGpuMat(Stream()), remapped,
        mapX.GetGpuMat(Stream()), mapY.GetGpuMat(Stream()),
        interpolation_method.getValue(), boarder_mode.getValue(), cv::Scalar(), Stream());
    undistorted_param.UpdateData(remapped, input_param.GetTimestamp(), _ctx);
    return true;
}


MO_REGISTER_CLASS(StereoBM)
MO_REGISTER_CLASS(StereoBilateralFilter)
MO_REGISTER_CLASS(StereoBeliefPropagation)
MO_REGISTER_CLASS(StereoConstantSpaceBP)
MO_REGISTER_CLASS(UndistortStereo)
