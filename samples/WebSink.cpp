#include "WebSink.hpp"
#include <EagleLib/Nodes/NodeInfo.hpp>
using namespace vclick;

WebSink::WebSink()
{
    h264_pass_through = mo::MetaObjectFactory::Instance()->Create("h264_pass_through");
    active_switch = h264_pass_through->GetParameter<bool>("active");
    moments.emplace_back(2, 0, 0);
    moments.emplace_back(0, 2, 0);
    moments.emplace_back(0, 0, 2);
}

bool WebSink::ProcessImpl()
{
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
        if (count == 0.0)
            continue;
        centroid /= count;
        for (int i = 0; i < moments.size(); ++i)
        {
            float value = moments[i].Evaluate(bb_mask, foreground_points, centroid);
            if (value > thresholds[i])
            {
                active_switch->UpdateData(true, point_cloud_param.GetTimestamp(), _ctx);
            }
        }
    }
    return false;
}
MO_REGISTER_CLASS(WebSink);