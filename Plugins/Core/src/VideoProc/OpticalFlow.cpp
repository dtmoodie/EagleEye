#include "OpticalFlow.h"
//#include "Aquila/Nodes/VideoProc/Tracking.hpp"
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
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
    if (input->getChannels() != 1)
    {
        cv::cuda::cvtColor(input->getGpuMat(stream()), greyImg[0], cv::COLOR_BGR2GRAY, 1, stream());
    }
    else
    {
        greyImg[0] = input->getGpuMat(stream());
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
        cv::cuda::pyrDown(pyramid[level - 1], pyramid[level], stream());
    }
}

bool DensePyrLKOpticalFlow::processImpl()
{
    if(window_size_param.modified() ||
        pyramid_levels_param.modified() ||
        iterations_param.modified() ||
        use_initial_flow_param.modified() ||
        opt_flow == nullptr)
    {

        opt_flow = cv::cuda::DensePyrLKOpticalFlow::create(
            cv::Size(window_size, window_size),
            pyramid_levels,
            iterations,
            use_initial_flow);

        window_size_param.modified(false);
        pyramid_levels_param.modified(false);
        iterations_param.modified(false);
        use_initial_flow_param.modified(false);
    }
    cv::cuda::GpuMat flow;
    if(prevGreyImg.empty())
    {
        prevGreyImg = greyImg;
        return true;
    }

    auto fn = PrepPyramid();

    opt_flow->calc(prevGreyImg, *image_pyramid, flow, stream());

    prevGreyImg = greyImg;
    flow_field_param.updateData(flow, fn, _ctx.get());
    return true;
}

bool SparsePyrLKOpticalFlow::processImpl()
{
    if (window_size_param.modified() ||
        pyramid_levels_param.modified() ||
        iterations_param.modified() ||
        use_initial_flow_param.modified() ||
        optFlow == nullptr)
    {
        optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(window_size, window_size),
            pyramid_levels,
            iterations,
            use_initial_flow);

        window_size_param.modified(false);
        pyramid_levels_param.modified(false);
        iterations_param.modified(false);
        use_initial_flow_param.modified(false);
    }
    auto ts = PrepPyramid();
    if(ts)
    {
        cv::cuda::GpuMat tracked_points, status, error;
        if(input_points_param.GetInput(ts - 1))
        {
            optFlow->calc(prevGreyImg, greyImg, input_points->getGpuMat(stream()), tracked_points, status, error, stream());
        }else
        {
            if(!prev_key_points.empty())
            {
                prev_key_points = input_points->getGpuMat(stream());
                return false;
            }else
            {
                optFlow->calc(prevGreyImg, greyImg, prev_key_points, tracked_points, status, error, stream());
            }
        }
        tracked_points_param.updateData(tracked_points, ts, _ctx.get());
        status_param.updateData(status, ts, _ctx.get());
        error_param.updateData(error, ts, _ctx.get());
        return true;
    }
    return false;
}

MO_REGISTER_CLASS(SparsePyrLKOpticalFlow)
MO_REGISTER_CLASS(DensePyrLKOpticalFlow)
