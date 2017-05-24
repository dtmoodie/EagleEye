#include "OpticalFlow.h"
//#include "Aquila/Nodes/VideoProc/Tracking.hpp"
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"
using namespace aq;
using namespace aq::Nodes;

#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cudaoptflow")
#endif

IPyrOpticalFlow::IPyrOpticalFlow()
{
    greyImg.resize(1);
}

size_t IPyrOpticalFlow::PrepPyramid()
{
    if (input->GetChannels() != 1)
    {
        cv::cuda::cvtColor(input->getGpuMat(Stream()), greyImg[0], cv::COLOR_BGR2GRAY, 1, Stream());
    }
    else
    {
        greyImg[0] = input->getGpuMat(Stream());
    }
    size_t fn;
    if (image_pyramid == nullptr)
    {
        image_pyramid = &greyImg;
        THROW(debug) << "Need to reimplement and redesign";

        fn= input_param.GetFrameNumber();
    }
    else
    {
        fn = image_pyramid_param.GetFrameNumber();
    }
    return fn;
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
    if(window_size_param._modified ||
        pyramid_levels_param._modified ||
        iterations_param._modified ||
        use_initial_flow_param._modified ||
        opt_flow == nullptr)
    {

        opt_flow = cv::cuda::DensePyrLKOpticalFlow::create(
            cv::Size(window_size, window_size),
            pyramid_levels,
            iterations,
            use_initial_flow);

        window_size_param._modified = false;
        pyramid_levels_param._modified = false;
        iterations_param._modified = false;
        use_initial_flow_param._modified = false;
    }
    cv::cuda::GpuMat flow;
    if(prevGreyImg.empty())
    {
        prevGreyImg = greyImg;
        return true;
    }

    auto fn = PrepPyramid();

    opt_flow->calc(prevGreyImg, *image_pyramid, flow, Stream());

    prevGreyImg = greyImg;
    flow_field_param.UpdateData(flow, fn, _ctx);
    return true;
}

bool SparsePyrLKOpticalFlow::ProcessImpl()
{
    if (window_size_param._modified ||
        pyramid_levels_param._modified ||
        iterations_param._modified ||
        use_initial_flow_param._modified ||
        optFlow == nullptr)
    {
        optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(window_size, window_size),
            pyramid_levels,
            iterations,
            use_initial_flow);

        window_size_param._modified = false;
        pyramid_levels_param._modified = false;
        iterations_param._modified = false;
        use_initial_flow_param._modified = false;
    }
    auto ts = PrepPyramid();
    if(ts)
    {
        cv::cuda::GpuMat tracked_points, status, error;
        if(input_points_param.GetInput(ts - 1))
        {
            optFlow->calc(prevGreyImg, greyImg, input_points->getGpuMat(Stream()), tracked_points, status, error, Stream());
        }else
        {
            if(!prev_key_points.empty())
            {
                prev_key_points = input_points->getGpuMat(Stream());
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
