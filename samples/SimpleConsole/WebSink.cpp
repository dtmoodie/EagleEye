#ifdef HAVE_WT
#include "WebSink.hpp"
#include <Aquila/Nodes/NodeInfo.hpp>
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"
using namespace vclick;

WebSink::WebSink():
    raw_bandwidth_mean(boost::accumulators::tag::rolling_window::window_size = 100),
    throttled_bandwidth_mean(boost::accumulators::tag::rolling_window::window_size = 100)
{
    h264_pass_through = mo::MetaObjectFactory::Instance()->Create("h264_pass_through");
    active_switch = h264_pass_through->GetParameter<bool>("active");
    moments.emplace_back(2, 0, 0);
    moments.emplace_back(0, 2, 0);
    moments.emplace_back(0, 0, 2);
    last_keyframe_time = boost::posix_time::microsec_clock::universal_time();
}
void WebSink::SetContext(mo::Context* ctx, bool overwrite)
{
    Node::SetContext(ctx, overwrite);
    h264_pass_through->SetContext(ctx, overwrite);
}
std::vector<mo::IParameter*> WebSink::GetParameters(const std::string& filter) const
{
    auto h264_params = h264_pass_through->GetParameters();
    auto my_params = Node::GetParameters();
    my_params.insert(my_params.end(), h264_params.begin(), h264_params.end());
    return my_params;
}

bool WebSink::ProcessImpl()
{
    if(moments.empty())
    {
        moments.emplace_back(2, 0, 0);
        moments.emplace_back(0, 2, 0);
        moments.emplace_back(0, 0, 2);
        thresholds.push_back(0.1f);
        thresholds.push_back(0.1f);
        thresholds.push_back(0.1f);
    }
    if(foreground_mask->empty() || point_cloud->empty())
        return false;
    cv::Mat mask = foreground_mask->GetMat(Stream());
    cv::Mat ptCloud = point_cloud->GetMat(Stream());
    int num_points = cv::countNonZero(mask);
    cv::Mat foreground_points(1, num_points, CV_32FC3);
    int count = 0;
    for (int i = 0; i < ptCloud.rows; ++i)
    {
        for (int j = 0; j < ptCloud.cols; ++j)
        {
            if (mask.at<uchar>(i, j))
            {
                foreground_points.at<cv::Vec3f>(count) = ptCloud.at<cv::Vec3f>(i, j);
                ++count;
            }
        }
    }
    foreground_points_param.UpdateData(foreground_points, point_cloud_param.GetTimestamp(), _ctx);
    bool activated = force_active;
    for (auto& bb : bounding_boxes)
    {
        cv::Mat bb_mask = bb.Contains(foreground_points);
        cv::Vec3f centroid(0, 0, 0);
        uchar* mask_ptr = bb_mask.ptr<uchar>();
        float count = 0;
        for (int i = 0; i < foreground_points.cols; ++i)
        {
            if (mask_ptr[i])
            {
                centroid += foreground_points.at<cv::Vec3f>(i);
                ++count;
            }
        }
        if (count < min_points)
            continue;
        centroid /= count;
        
        if(thresholds.size() < moments.size())
        {
            while(thresholds.size() != moments.size())
            {
                thresholds.push_back(0.1);
            }
        }
        for (int i = 0; i < moments.size(); ++i)
        {
            float value = moments[i].Evaluate(bb_mask, foreground_points, centroid);
            if (value > thresholds[i])
            {
                activated = true;
            }
        }
    }
    active_switch->UpdateData(activated, point_cloud_param.GetTimestamp(), _ctx);
    
    h264_pass_through->Process();
    
    auto current_time = boost::posix_time::microsec_clock::universal_time();
    if(boost::posix_time::time_duration(current_time - last_keyframe_time).total_milliseconds() > heartbeat_ms || activated == true)
    {
        if(jpeg_buffer && !jpeg_buffer->empty())
        {
            last_keyframe_time = current_time;
            output_jpeg_param.UpdateData(*jpeg_buffer, jpeg_buffer_param.GetTimestamp(), _ctx);
        }
        if(raw_image && !raw_image->empty())
        {
            last_keyframe_time = current_time;
            output_image_param.UpdateData(*raw_image, raw_image_param.GetTimestamp(), _ctx);
        }
        if(jpeg_buffer)
        {
            throttled_bandwidth_mean(double(jpeg_buffer->size().area()));
        }
    }else
    {
        throttled_bandwidth_mean(0.0);
    }
    if(jpeg_buffer)
    {
        raw_bandwidth_mean(jpeg_buffer->size().area());
    }
    raw_bandwidth_param.UpdateData(boost::accumulators::rolling_mean(raw_bandwidth_mean), background_model_param.GetTimestamp(), _ctx);
    throttled_bandwidth_param.UpdateData(boost::accumulators::rolling_mean(throttled_bandwidth_mean), background_model_param.GetTimestamp(), _ctx);
    return true;
}
MO_REGISTER_CLASS(WebSink);
#endif
