#include "FeatureDetection.h"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"

using namespace aq;
using namespace aq::Nodes;


bool GoodFeaturesToTrack::processImpl()
{
    cv::cuda::GpuMat grey;
    if(input->getChannels() != 1)
    {
        cv::cuda::cvtColor(input->getGpuMat(stream()), grey, cv::COLOR_BGR2GRAY, 0, stream());
    }else
    {
        grey = input->getGpuMat(stream());
    }
    if(!detector || max_corners_param.modified() || quality_level_param.modified() || min_distance_param.modified() || block_size_param.modified()
        || use_harris_param.modified()/* || harris_K_param.modified()*/)
    {
        detector = cv::cuda::createGoodFeaturesToTrackDetector(input->getDepth(), max_corners, quality_level, min_distance, block_size, use_harris);
        max_corners_param.modified(false);
        quality_level_param.modified()  = false;
        min_distance_param.modified(false);
        block_size_param.modified(false);
        use_harris_param.modified(false);
    }
    cv::cuda::GpuMat keypoints;
    if(mask)
    {
        detector->detect(grey, keypoints, mask->getGpuMat(stream()), stream());
    }
    else
    {
        detector->detect(grey, keypoints, cv::noArray(), stream());
    }
    key_points_param.updateData(keypoints, input_param.getTimestamp(), _ctx.get());
    num_corners_param.updateData(keypoints.cols, input_param.getTimestamp(), _ctx.get());
    return true;
}



bool FastFeatureDetector::processImpl()
{
    if(threshold_param.modified() ||
        use_nonmax_suppression_param.modified() ||
        fast_type_param.modified() ||
        max_points_param.modified() ||
        detector == nullptr)
    {
        detector = cv::cuda::FastFeatureDetector::create(threshold, use_nonmax_suppression, fast_type.getValue(), max_points);
    }
    cv::cuda::GpuMat keypoints;
    if(mask)
    {
        detector->detectAsync(input->getGpuMat(stream()), keypoints, mask->getGpuMat(stream()), stream());
    }else
    {
        detector->detectAsync(input->getGpuMat(stream()), keypoints, cv::noArray(), stream());
    }
    if(!keypoints.empty())
    {
        keypoints_param.updateData(keypoints, input_param.getTimestamp(), _ctx.get());
    }
    return true;
}


/// *****************************************************************************************
/// *****************************************************************************************
/// *****************************************************************************************




bool ORBFeatureDetector::processImpl()
{
    if(num_features_param.modified() || scale_factor_param.modified() ||
        num_levels_param.modified() || edge_threshold_param.modified() ||
        first_level_param.modified() || WTA_K_param.modified() || score_type_param.modified() ||
        patch_size_param.modified() || fast_threshold_param.modified() || blur_for_descriptor_param.modified() ||
        detector == nullptr)
    {
        detector = cv::cuda::ORB::create(num_features, scale_factor, num_levels, edge_threshold, first_level,
            WTA_K, score_type.getValue(), patch_size, fast_threshold, blur_for_descriptor);
        num_features_param.modified(false);
        scale_factor_param.modified(false);
        num_levels_param.modified(false);
        edge_threshold_param.modified(false);
        first_level_param.modified(false);
        WTA_K_param.modified(false);
        score_type_param.modified(false);
        patch_size_param.modified(false);
        fast_threshold_param.modified(false);
        blur_for_descriptor_param.modified(false);
    }
    cv::cuda::GpuMat keypoints;
    cv::cuda::GpuMat descriptors;
    if(mask)
    {
        detector->detectAndComputeAsync(input->getGpuMat(stream()), mask->getGpuMat(stream()), keypoints, descriptors, false, stream());
    }else
    {
        detector->detectAndComputeAsync(input->getGpuMat(stream()), cv::noArray(), keypoints, descriptors, false, stream());
    }
    keypoints_param.updateData(keypoints, input_param.getTimestamp(), _ctx.get());
    descriptors_param.updateData(descriptors, input_param.getTimestamp(), _ctx.get());
    return true;
}




bool CornerHarris::processImpl()
{
    if(block_size_param.modified() || sobel_aperature_size_param.modified() || harris_free_parameter_param.modified() || detector == nullptr)
    {
        detector = cv::cuda::createHarrisCorner(input->GetType(), block_size, sobel_aperature_size, harris_free_parameter);
    }
    cv::cuda::GpuMat score;
    detector->compute(input->getGpuMat(stream()), score, stream());
    score_param.updateData(score, input_param.getTimestamp(), _ctx.get());
    return true;
}



bool CornerMinEigenValue::processImpl()
{
    if (block_size_param.modified() || sobel_aperature_size_param.modified() || harris_free_parameter_param.modified() || detector == nullptr)
    {
        detector = cv::cuda::createMinEigenValCorner(input->GetType(), block_size, sobel_aperature_size, harris_free_parameter);
    }
    cv::cuda::GpuMat score;
    detector->compute(input->getGpuMat(stream()), score, stream());
    score_param.updateData(score, input_param.getTimestamp(), _ctx.get());
    return true;
}



MO_REGISTER_CLASS(GoodFeaturesToTrack)
MO_REGISTER_CLASS(ORBFeatureDetector)
MO_REGISTER_CLASS(FastFeatureDetector)

MO_REGISTER_CLASS(CornerHarris)


