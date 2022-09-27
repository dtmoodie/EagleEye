#include "Stereo.h"

#include <ct/reflect/cerealize.hpp>

#include <Aquila/nodes/NodeInfo.hpp>

#include "Aquila/rcc/external_includes/cv_cudawarping.hpp"
#include "Aquila/rcc/external_includes/cv_imgproc.hpp"
#include <Aquila/rcc/external_includes/cv_cudastereo.hpp>

namespace aqdev
{
    bool StereoBM::processImpl(aq::CVStream& stream)
    {
        if (!stereoBM || num_disparities_param.getModified() || block_size_param.getModified())
        {
            stereoBM = cv::cuda::createStereoBM(num_disparities, block_size);
            block_size_param.setModified(false);
            num_disparities_param.setModified(false);
        }
        if ((left_image->size() != right_image->size()).all())
        {
            MO_LOG(debug, "Images are of mismatched size {} != {}", left_image->size(), right_image->size());
            return false;
        }

        cv::cuda::GpuMat left = this->left_image->getGpuMat(&stream);
        cv::cuda::GpuMat right = this->right_image->getGpuMat(&stream);
        cv::cuda::GpuMat disparity;
        cv::cuda::Stream& cv_stream = stream.getCVStream();
        MO_ASSERT(cv_stream != nullptr);

        stereoBM->compute(left, right, disparity, cv_stream);

        this->disparity.publish(disparity, mo::tags::param = &left_image_param);
        return true;
    }

    bool StereoBeliefPropagation::processImpl(aq::CVStream& stream)
    {
        if (bp == nullptr || num_iters_param.getModified() || num_disparities_param.getModified() ||
                num_levels_param.getModified(),
            message_type_param.getModified())
        {
            bp = cv::cuda::createStereoBeliefPropagation(
                num_disparities, num_iters, num_levels, message_type.getValue());

            num_iters_param.setModified(true);
            num_disparities_param.setModified(true);
            num_levels_param.setModified(true);
            message_type_param.setModified(true);
        }

        cv::cuda::GpuMat left = this->left_image->getGpuMat(&stream);
        cv::cuda::GpuMat right = this->right_image->getGpuMat(&stream);
        cv::cuda::Stream& cv_stream = stream.getCVStream();
        cv::cuda::GpuMat disparity;

        this->bp->compute(left, right, disparity, cv_stream);
        this->disparity.publish(disparity, mo::tags::param = &left_image_param);
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
            //createStereoConstantSpaceBP(int ndisp = 128, int iters = 8, int levels = 4, int nr_plane = 4, int msg_type
    = CV_32F); addInputParam<cv::cuda::GpuMat>("Left image"); addInputParam<cv::cuda::GpuMat>("Right image"); csbp =
    cv::cuda::createStereoConstantSpaceBP(); }else
        {
            _parameters[0]->changed = true;
        }
    }*/
    bool StereoConstantSpaceBP::processImpl(aq::CVStream& stream)
    {
        if (num_levels_param.getModified() || nr_plane_param.getModified() || num_disparities_param.getModified() ||
            num_iterations_param.getModified() || !csbp)
        {
            csbp = cv::cuda::createStereoConstantSpaceBP(
                num_disparities, num_iterations, num_levels, nr_plane, message_type.getValue());
            num_levels_param.setModified(false);
            nr_plane_param.setModified(false);
            num_disparities_param.setModified(false);
            num_iterations_param.setModified(false);
        }

        cv::cuda::GpuMat left = this->left_image->getGpuMat(&stream);
        cv::cuda::GpuMat right = this->right_image->getGpuMat(&stream);
        cv::cuda::GpuMat disparity;
        cv::cuda::Stream cv_stream = stream.getCVStream();

        csbp->compute(left, right, disparity, cv_stream);

        this->disparity.publish(disparity, mo::tags::param = &left_image_param);
        return true;
    }

    bool UndistortStereo::processImpl(aq::CVStream& stream)
    {
        const bool updated_intrinsic_params = camera_matrix_param.hasNewData() || distortion_matrix_param.hasNewData();
        const bool updated_extrinsic_params =
            rotation_matrix_param.hasNewData() || projection_matrix_param.hasNewData();
        const bool warp_maps_need_initialization = (this->m_map_x.empty() || this->m_map_y.empty());
        cv::cuda::Stream& cvstream = stream.getCVStream();

        if (updated_intrinsic_params || updated_extrinsic_params || warp_maps_need_initialization)
        {
            auto size = input->size();
            cv::Size sz(size(0), size(1));
            cv::Mat X, Y;
            cv::initUndistortRectifyMap(
                *camera_matrix, *distortion_matrix, *rotation_matrix, *projection_matrix, sz, CV_32FC1, X, Y);

            m_map_x.upload(X, cvstream);
            m_map_y.upload(Y, cvstream);

            this->mapX.publish(X);
            this->mapY.publish(Y);
        }

        cv::cuda::GpuMat remapped;
        cv::cuda::GpuMat in = input->getGpuMat(&stream);

        cv::cuda::remap(input->getGpuMat(&stream),
                        remapped,
                        this->m_map_x,
                        this->m_map_y,
                        interpolation_method.getValue(),
                        boarder_mode.getValue(),
                        cv::Scalar(),
                        cvstream);
        undistorted.publish(remapped, mo::tags::param = &this->input_param);
        return true;
    }

} // namespace aqdev

using namespace aqdev;
MO_REGISTER_CLASS(StereoBM)
MO_REGISTER_CLASS(StereoBilateralFilter)
MO_REGISTER_CLASS(StereoBeliefPropagation)
MO_REGISTER_CLASS(StereoConstantSpaceBP)
MO_REGISTER_CLASS(UndistortStereo)
