
#include "Registration.h"
#include <thrust/transform.h>
#include <opencv2/core/cuda_stream_accessor.hpp>



using namespace aq;
using namespace aq::Nodes;
/*
void register_to_reference::NodeInit(bool firstInit)
{
    d_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    d_orb = cv::cuda::ORB::create(500, 1.2, 8, 31, 0, 2, cv::cuda::ORB::HARRIS_SCORE, 31, 20, true);

    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Reference image");
        updateParameter<int>("Num features", 500);
        updateParameter<float>("Scale factor", 1.2);
        updateParameter<int>("Pyramid levels", 8);
        updateParameter<int>("Edge threshold", 31);
        updateParameter<int>("first level", 0);
        updateParameter<int>("WTA_K", 2);
        updateParameter<int>("Patch size", 31);
        updateParameter<int>("Fast threshold", 20);
        updateParameter<bool>("Blur for descriptors", true);
    }
}
void register_to_reference::doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    for(int i = 1; i < 10; ++i)
    {
        if(_parameters[i]->changed)
        {
            d_orb = cv::cuda::ORB::create(
                *getParameter<int>(1)->Data(),
                *getParameter<float>(2)->Data(),
                *getParameter<int>(3)->Data(),
                *getParameter<int>(4)->Data(),
                *getParameter<int>(5)->Data(),
                *getParameter<int>(6)->Data(),
                cv::cuda::ORB::HARRIS_SCORE,
                *getParameter<int>(7)->Data(),
                *getParameter<int>(8)->Data(),
                *getParameter<bool>(9)->Data());
            for(int i = 1; i < 10; ++i)
                _parameters[i]->changed = false;
            break;
        }
    }
    if(_parameters[0]->changed || ref_descriptors.empty())
    {
        auto ref_mat = getParameter<cv::cuda::GpuMat>(0)->Data();
        if(ref_mat && ! ref_mat->empty())
        {
            d_reference_original = *ref_mat;
            if(d_reference_original.channels() == 3)
                cv::cuda::cvtColor(d_reference_original, d_reference_grey, cv::COLOR_BGR2GRAY, 0, stream);
            else
                d_reference_grey = d_reference_original;
            if(d_reference_grey.depth() != CV_8UC1)
            {
                cv::cuda::GpuMat tmp;
                cv::cuda::normalize(d_reference_grey, tmp, 0, 255, cv::NORM_MINMAX, CV_8U, cv::noArray(), stream);
                d_reference_grey = tmp;
            }
            cv::cuda::GpuMat temp_ref_keypoints;
            d_orb->detectAndComputeAsync(d_reference_grey, cv::noArray(), temp_ref_keypoints, ref_descriptors, false, stream);
            temp_ref_keypoints.download(ref_keypoints, stream);
        }
        _parameters[0]->changed = false;
    }
    if(ref_descriptors.empty() || ref_keypoints.empty() || d_reference_original.empty())
        return;
    auto gpu_mat = input.getGpuMat(stream);
    cv::cuda::GpuMat grey;
    if(gpu_mat.channels() == 3)
        cv::cuda::cvtColor(gpu_mat, grey, cv::COLOR_BGR2GRAY, 0, stream);
    else
        grey = gpu_mat;
    if(grey.depth() != CV_8U)
    {
        cv::cuda::GpuMat tmp;
        cv::cuda::normalize(grey, tmp, 0, 255, cv::NORM_MINMAX, CV_8U, cv::noArray(), stream);
        grey = tmp;
    }
    cv::cuda::GpuMat descriptors, keypoints, matches;
    cv::Mat h_keypoints;
    d_orb->detectAndComputeAsync(grey, cv::noArray(), keypoints, descriptors, false, stream);
    d_matcher->matchAsync(descriptors, ref_descriptors, matches, cv::noArray(), stream);
    cv::Mat h_matches;
    matches.download(h_matches, stream);
    keypoints.download(h_keypoints, stream);
    stream.waitForCompletion();
    int* idx = h_matches.ptr<int>(0);
    float* dist = h_matches.ptr<float>(1);
    std::vector<cv::Point2f> ref_matched_points;
    std::vector<cv::Point2f> input_matched_points;
    
    cv::Mat dist_mat(1, h_matches.cols, CV_32F, dist);
    cv::Scalar mean, stddev;
    cv::meanStdDev(dist_mat, mean, stddev);
    float threshold = mean.val[0] + stddev.val[0]*2;
    std::vector<cv::DMatch> _matches;
    d_matcher->matchConvert(matches,_matches);
    for(int i = 0; i < h_matches.cols; ++i)
    {
        if(dist[i] < threshold)
        {
            ref_matched_points.push_back(cv::Point2f(ref_keypoints.at<float>(0, idx[i]), ref_keypoints.at<float>(1, idx[i])));
            input_matched_points.push_back(cv::Point2f(h_keypoints.at<float>(0, i), h_keypoints.at<float>(1,i)));
        }
    }
    cv::Mat point_mask;
    cv::Mat H = cv::findHomography(ref_matched_points, input_matched_points, cv::RANSAC, 3.0, point_mask, 2000, 0.995);
    cv::cuda::GpuMat mask(gpu_mat.size(), CV_32F);
    mask.setTo(cv::Scalar(1.0), stream);
    cv::cuda::GpuMat warped_input, warped_input_mask;
    cv::cuda::warpPerspective(input.getGpuMat(stream), warped_input, H, gpu_mat.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, 0, cv::Scalar(), stream);
    cv::cuda::warpPerspective(mask, warped_input_mask, H, gpu_mat.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, 0, cv::Scalar(), stream);
    cv::cuda::bitwise_and(mask, warped_input_mask, warped_input_mask, cv::noArray(), stream);
    cv::cuda::GpuMat blended;
    cv::cuda::blendLinear(warped_input, d_reference_original, warped_input_mask, mask,blended, stream);
    updateParameter("Overlay image", blended);
    updateParameter("Input wrt reference", warped_input, &stream);
    updateParameter("Input mask wrt reference", warped_input_mask, &stream);
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(register_to_reference);*/