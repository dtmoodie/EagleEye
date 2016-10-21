#include "OpticalFlow.h"
#include "EagleLib/nodes/VideoProc/Tracking.hpp"
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudawarping.hpp>

using namespace EagleLib;
using namespace EagleLib::Nodes;

#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cudaoptflow")
#endif

IPyrOpticalFlow::IPyrOpticalFlow()
{
    greyImg.resize(1);
}

long long IPyrOpticalFlow::PrepPyramid()
{
    if (input->GetChannels() != 1)
    {
        cv::cuda::cvtColor(input->GetGpuMat(Stream()), greyImg[0], cv::COLOR_BGR2GRAY, 1, Stream());
    }
    else
    {
        greyImg[0] = input->GetGpuMat(Stream());
    }
    long long timestamp;
    if (image_pyramid == nullptr)
    {
        image_pyramid = &greyImg;
        build_pyramid(*image_pyramid);
        timestamp = input_param.GetTimestamp();
    }
    else
    {
        timestamp = image_pyramid_param.GetTimestamp();
    }

    if (prevGreyImg.empty())
    {
        prevGreyImg = greyImg;
        return -1;
    }
    return timestamp;
}

void IPyrOpticalFlow::build_pyramid(std::vector<cv::cuda::GpuMat>& pyramid)
{
    CV_Assert(pyramid.size());
    CV_Assert(!pyramid[0].empty());
    pyramid.resize(pyramid_levels);
    for (int level = 1; level < pyramid_levels; ++level)
    {
        cv::cuda::pyrDown(pyramid[level - 1], pyramid[level], Stream());
    }
}

bool DensePyrLKOpticalFlow::ProcessImpl()
{
    if(window_size_param.modified ||
        pyramid_levels_param.modified ||
        iterations_param.modified ||
        use_initial_flow_param.modified ||
        opt_flow == nullptr)
    {
        
        opt_flow = cv::cuda::DensePyrLKOpticalFlow::create(
            cv::Size(window_size, window_size), 
            pyramid_levels,
            iterations,
            use_initial_flow);

        window_size_param.modified = false;
        pyramid_levels_param.modified = false;
        iterations_param.modified = false;
        use_initial_flow_param.modified = false;
    }
    cv::cuda::GpuMat flow;
    long long ts = PrepPyramid();
    if(ts != -1)
    {
        opt_flow->calc(prevGreyImg, *image_pyramid, flow, Stream());

        prevGreyImg = greyImg;
        flow_field_param.UpdateData(flow, ts, _ctx);
        return true;
    }
    return false;
}

bool SparsePyrLKOpticalFlow::ProcessImpl()
{
    if (window_size_param.modified ||
        pyramid_levels_param.modified ||
        iterations_param.modified ||
        use_initial_flow_param.modified ||
        optFlow == nullptr)
    {
        optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(window_size, window_size),
            pyramid_levels,
            iterations,
            use_initial_flow);

        window_size_param.modified = false;
        pyramid_levels_param.modified = false;
        iterations_param.modified = false;
        use_initial_flow_param.modified = false;
    }
    long long ts = PrepPyramid();
    if(ts != -1)
    {
        cv::cuda::GpuMat tracked_points, status, error;
        if(input_points_param.GetInput(ts - 1))
        {
            optFlow->calc(prevGreyImg, greyImg, input_points->GetGpuMat(Stream()), tracked_points, status, error, Stream());
        }else
        {
            if(!prev_key_points.empty())
            {
                prev_key_points = input_points->GetGpuMat(Stream());
                return false;
            }else
            {
                optFlow->calc(prevGreyImg, greyImg, prev_key_points, tracked_points, status, error, Stream());
            }
        }
        tracked_points_param.UpdateData(tracked_points, ts, _ctx);
        status_param.UpdateData(status, ts, _ctx);
        error_param.UpdateData(error, ts, _ctx);
        return true;
    }
    return false;
}

MO_REGISTER_CLASS(SparsePyrLKOpticalFlow)
MO_REGISTER_CLASS(DensePyrLKOpticalFlow)