#include "OpticalFlow.h"
#include <MetaObject/core/metaobject_config.hpp>
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>

namespace aq
{
namespace nodes
{

template<>
bool PyrLKLandmarkTracker::processImpl(mo::Context* ctx)
{
    cv::Mat in = input->getMat(ctx);
    cv::Mat gray;
    if(in.channels() != 1)
    {
        cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    }else
    {
        gray = in;
    }

    //std::vector<cv::Mat> pyr;
    //cv::buildOpticalFlowPyramid(gray, pyr, cv::Size(window_size, window_size), pyramid_levels);
    if(m_prev_pyramid.empty())
    {
        m_prev_pyramid = TS<SyncedMemory>(input_param.getTimestamp(), input_param.getFrameNumber(), gray);
        return true;
    }else
    {
        const auto fn = input_param.getFrameNumber();
        std::shared_ptr<const LandmarkDetectionSet> dets;
        mo::OptionalTime_t ts;
        MO_ASSERT(detections_param.checkFlags(mo::ParamFlags::RequestBuffered_e));
        if(detections_param.getData(dets, fn - 1, ctx, &ts))
        {
            std::vector<cv::Point2f> points;
            std::vector<size_t> size;
            for(size_t det_id = 0; det_id < dets->size(); ++det_id)
            {
                cv::Mat pts =(*dets)[det_id].landmark_keypoints.getMat(ctx);
                size.push_back(pts.cols);
                for(int i = 0; i < pts.cols; ++i)
                {
                    points.push_back(pts.at<cv::Point2f>(i));
                }
            }
            output.setCatSet(dets->getCatSet());
            if(!points.empty())
            {
                cv::Mat_<cv::Point2f> tracked_points;
                cv::Mat status, error;
                cv::calcOpticalFlowPyrLK(m_prev_pyramid.getMat(ctx), gray, points,tracked_points, status, error, cv::Size(window_size, window_size), pyramid_levels);

                size_t start = 0;
                for(size_t i = 0; i < dets->size(); ++i)
                {
                    LandmarkDetection out_det((*dets)[i]);
                    out_det.landmark_keypoints = aq::SyncedMemory(tracked_points.colRange(start, start + size[i]));
                    start += size[i];
                    output.push_back(std::move(out_det));
                }
            }

            output_param.emitUpdate(input_param);
            return true;
        }else
        {
            return false;
        }
    }
}

bool PyrLKLandmarkTracker::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

template<>
bool PyrLKLandmarkTracker::processImpl(mo::CvContext* ctx)
{
    if((input->getSyncState(0) < input->DEVICE_UPDATED))
    {
        return processImpl(static_cast<mo::Context*>(ctx));
    }
    // TODO
    return false;
}

}
}

using namespace aq::nodes;

MO_REGISTER_CLASS(PyrLKLandmarkTracker)
