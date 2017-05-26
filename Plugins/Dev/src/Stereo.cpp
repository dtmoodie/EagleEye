#include "Stereo.h"
#include "Aquila/rcc/external_includes/cv_imgproc.hpp"
#include "Aquila/rcc/external_includes/cv_cudawarping.hpp"

#if _WIN32
    #if _DEBUG
        RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300d.lib")
    #else
        RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300.lib")
    #endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudastereo")
#endif



using namespace aq;
using namespace aq::Nodes;


bool StereoBM::processImpl()
{
    if(!stereoBM || num_disparities_param.modified() || block_size_param.modified())
    {
        stereoBM = cv::cuda::createStereoBM(num_disparities, block_size);
        block_size_param.modified(false);
        num_disparities_param.modified(false);
    }
    if (left_image->getSize() != right_image->getSize())
    {
        //log(Error, "Images are of mismatched size");
        LOG(debug) << "Images are of mismatched size";
        return false;
    }
    cv::cuda::GpuMat disparity;
    stereoBM->compute(left_image->getGpuMat(stream()), right_image->getGpuMat(stream()),disparity, stream());
    this->disparity_param.updateData(disparity, left_image_param.getTimestamp(), _ctx.get());
    return true;
}

bool StereoBeliefPropagation::processImpl()
{
    if(bp == nullptr ||
        num_iters_param.modified() ||
        num_disparities_param.modified() ||
        num_levels_param.modified(),
        message_type_param.modified())
    {
        bp = cv::cuda::createStereoBeliefPropagation(num_disparities, num_iters, num_levels, message_type.getValue());

        num_iters_param.modified(true);
        num_disparities_param.modified(true);
        num_levels_param.modified(true);
        message_type_param.modified(true);
    }
    cv::cuda::GpuMat disparity;
    bp->compute(left_image->getGpuMat(stream()), right_image->getGpuMat(stream()), disparity, stream());
    disparity_param.updateData(disparity, left_image_param.getTimestamp(), _ctx.get());
    return true;
}



/*void StereoConstantSpaceBP::nodeInit(bool firstInit)
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
        addInputParam<cv::cuda::GpuMat>("Left image");
        addInputParam<cv::cuda::GpuMat>("Right image");
        csbp = cv::cuda::createStereoConstantSpaceBP();
    }else
    {
        _parameters[0]->changed = true;
    }
}*/
bool StereoConstantSpaceBP::processImpl()
{
    if(num_levels_param.modified() || nr_plane_param.modified() ||
        num_disparities_param.modified() || num_iterations_param.modified() || !csbp)
    {
        csbp = cv::cuda::createStereoConstantSpaceBP(
            num_disparities,
            num_iterations,
            num_levels, nr_plane, message_type.getValue());
        num_levels_param.modified(false);
        nr_plane_param.modified(false);
        num_disparities_param.modified(false);
        num_iterations_param.modified(false);
    }
    cv::cuda::GpuMat disparity;
    csbp->compute(left_image->getGpuMat(stream()), right_image->getGpuMat(stream()),disparity, stream());
    this->disparity_param.updateData(disparity, left_image_param.getTimestamp(), _ctx.get());
    return true;
}

bool UndistortStereo::processImpl()
{
    if(camera_matrix_param.modified() || distortion_matrix_param.modified() ||
        rotation_matrix_param.modified() || projection_matrix_param.modified())
    {
        cv::Mat X, Y;
        cv::initUndistortRectifyMap(*camera_matrix, *distortion_matrix,
            *rotation_matrix, *projection_matrix, input->getSize(), CV_32FC1, X, Y);
        mapX_param.updateData(X, -1, _ctx.get());
        mapY_param.updateData(Y, -1, _ctx.get());
    }
    cv::cuda::GpuMat remapped;
    cv::cuda::remap(input->getGpuMat(stream()), remapped,
        mapX.getGpuMat(stream()), mapY.getGpuMat(stream()),
        interpolation_method.getValue(), boarder_mode.getValue(), cv::Scalar(), stream());
    undistorted_param.updateData(remapped, input_param.getTimestamp(), _ctx.get());
    return true;
}


MO_REGISTER_CLASS(StereoBM)
MO_REGISTER_CLASS(StereoBilateralFilter)
MO_REGISTER_CLASS(StereoBeliefPropagation)
MO_REGISTER_CLASS(StereoConstantSpaceBP)
MO_REGISTER_CLASS(UndistortStereo)
