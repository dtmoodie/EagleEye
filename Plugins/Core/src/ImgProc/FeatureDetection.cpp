#include "FeatureDetection.h"




using namespace EagleLib;
using namespace EagleLib::Nodes;


bool GoodFeaturesToTrack::ProcessImpl()
{
    cv::cuda::GpuMat grey;
    if(input->GetChannels() != 1)
    {
        cv::cuda::cvtColor(input->GetGpuMat(Stream()), grey, cv::COLOR_BGR2GRAY, 0, Stream());
    }else
    {
        grey = input->GetGpuMat(Stream());
    }
    if(!detector || max_corners_param.modified || quality_level_param.modified || min_distance_param.modified || block_size_param.modified
        || use_harris_param.modified/* || harris_K_param.modified*/)
    {
        detector = cv::cuda::createGoodFeaturesToTrackDetector(input->GetDepth(), max_corners, quality_level, min_distance, block_size, use_harris);
        max_corners_param.modified = false;
        quality_level_param.modified  = false;
        min_distance_param.modified = false; 
        block_size_param.modified = false;
        use_harris_param.modified = false;
    }
    cv::cuda::GpuMat keypoints;
    if(mask)
    {
        detector->detect(grey, keypoints, mask->GetGpuMat(Stream()), Stream());
    }
    else
    {
        detector->detect(grey, keypoints, cv::noArray(), Stream());
    }
    key_points_param.UpdateData(keypoints, input_param.GetTimestamp(), _ctx);
    num_corners_param.UpdateData(keypoints.cols, input_param.GetTimestamp(), _ctx);
    return true;
}



bool FastFeatureDetector::ProcessImpl()
{
    if(threshold_param.modified ||
        use_nonmax_suppression_param.modified ||
        fast_type_param.modified ||
        max_points_param.modified ||
        detector == nullptr)
    {
        detector = cv::cuda::FastFeatureDetector::create(threshold, use_nonmax_suppression, fast_type.getValue(), max_points);
    }
    cv::cuda::GpuMat keypoints;
    if(mask)
    {
        detector->detectAsync(input->GetGpuMat(Stream()), keypoints, mask->GetGpuMat(Stream()), Stream());
    }else
    {
        detector->detectAsync(input->GetGpuMat(Stream()), keypoints, cv::noArray(), Stream());
    }
    if(!keypoints.empty())
    {
        keypoints_param.UpdateData(keypoints, input_param.GetTimestamp(), _ctx);
    }
    return true;
}


/// *****************************************************************************************
/// *****************************************************************************************
/// *****************************************************************************************




bool ORBFeatureDetector::ProcessImpl()
{
    if(num_features_param.modified || scale_factor_param.modified ||
        num_levels_param.modified || edge_threshold_param.modified ||
        first_level_param.modified || WTA_K_param.modified || score_type_param.modified ||
        patch_size_param.modified || fast_threshold_param.modified || blur_for_descriptor_param.modified ||
        detector == nullptr)
    {
        detector = cv::cuda::ORB::create(num_features, scale_factor, num_levels, edge_threshold, first_level,
            WTA_K, score_type.getValue(), patch_size, fast_threshold, blur_for_descriptor);
        num_features_param.modified = false;
        scale_factor_param.modified = false;
        num_levels_param.modified = false;
        edge_threshold_param.modified = false;
        first_level_param.modified = false;
        WTA_K_param.modified = false;
        score_type_param.modified = false;
        patch_size_param.modified = false;
        fast_threshold_param.modified = false;
        blur_for_descriptor_param.modified = false;
    }
    cv::cuda::GpuMat keypoints;
    cv::cuda::GpuMat descriptors;
    if(mask)
    {
        detector->detectAndComputeAsync(input->GetGpuMat(Stream()), mask->GetGpuMat(Stream()), keypoints, descriptors, false, Stream());
    }else
    {
        detector->detectAndComputeAsync(input->GetGpuMat(Stream()), cv::noArray(), keypoints, descriptors, false, Stream());
    }
    keypoints_param.UpdateData(keypoints, input_param.GetTimestamp(), _ctx);
    descriptors_param.UpdateData(descriptors, input_param.GetTimestamp(), _ctx);
    return true;
}


bool HistogramRange::ProcessImpl()
{
    if(lower_bound_param.modified || upper_bound_param.modified || bins_param.modified)
    {
        updateLevels(input->GetDepth());
        lower_bound_param.modified = false;
        upper_bound_param.modified = false;
        bins_param.modified = false;
    }
    if(input->GetChannels() == 1 || input->GetChannels() == 4)
    {
        cv::cuda::GpuMat hist;
        cv::cuda::histRange(input->GetGpuMat(Stream()), hist, levels.GetGpuMat(Stream()), Stream());
        histogram_param.UpdateData(hist, input_param.GetTimestamp(), _ctx);
        return true;
    }
    return false;
}


void HistogramRange::updateLevels(int type)
{
    cv::Mat h_mat;
    if(type == CV_32F)
        h_mat = cv::Mat(1, bins, CV_32F);
    else
        h_mat = cv::Mat(1, bins, CV_32S);
    double step = (upper_bound - lower_bound) / double(bins);

    double val = lower_bound;
    for(int i = 0; i < bins; ++i, val += step)
    {
        if(type == CV_32F)
            h_mat.at<float>(i) = val;
        if(type == CV_8U)
            h_mat.at<int>(i) = val;
    }
    levels_param.UpdateData(h_mat);
}

bool CornerHarris::ProcessImpl()
{
    if(block_size_param.modified || sobel_aperature_size_param.modified || harris_free_parameter_param.modified || detector == nullptr)
    {
        detector = cv::cuda::createHarrisCorner(input->GetType(), block_size, sobel_aperature_size, harris_free_parameter);
    }
    cv::cuda::GpuMat score;
    detector->compute(input->GetGpuMat(Stream()), score, Stream());
    score_param.UpdateData(score, input_param.GetTimestamp(), _ctx);
    return true;
}



bool CornerMinEigenValue::ProcessImpl()
{
    if (block_size_param.modified || sobel_aperature_size_param.modified || harris_free_parameter_param.modified || detector == nullptr)
    {
        detector = cv::cuda::createMinEigenValCorner(input->GetType(), block_size, sobel_aperature_size, harris_free_parameter);
    }
    cv::cuda::GpuMat score;
    detector->compute(input->GetGpuMat(Stream()), score, Stream());
    score_param.UpdateData(score, input_param.GetTimestamp(), _ctx);
    return true;
}



MO_REGISTER_CLASS(GoodFeaturesToTrack);
MO_REGISTER_CLASS(ORBFeatureDetector);
MO_REGISTER_CLASS(FastFeatureDetector);
MO_REGISTER_CLASS(HistogramRange);
MO_REGISTER_CLASS(CornerHarris);


