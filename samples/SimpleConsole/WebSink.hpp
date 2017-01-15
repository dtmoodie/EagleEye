#ifdef HAVE_WT
#pragma once
#include <EagleLib/Nodes/Node.h>
#include <boost/date_time.hpp>
#include "BoundingBox.hpp"
#include "Moment.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
namespace vclick
{
    class WebSink : public EagleLib::Nodes::Node
    {
    public:
        WebSink();
        MO_DERIVE(WebSink, EagleLib::Nodes::Node)
            INPUT(EagleLib::SyncedMemory, background_model, nullptr);
            INPUT(EagleLib::SyncedMemory, foreground_mask, nullptr);
            INPUT(EagleLib::SyncedMemory, point_cloud, nullptr);
            OPTIONAL_INPUT(cv::Mat, jpeg_buffer, nullptr);
            OPTIONAL_INPUT(EagleLib::SyncedMemory, raw_image, nullptr);
            
            OUTPUT(cv::Mat, foreground_points, cv::Mat());
            OUTPUT(cv::Mat, output_jpeg, cv::Mat());
            OUTPUT(EagleLib::SyncedMemory, output_image, EagleLib::SyncedMemory());
            OUTPUT(double, throttled_bandwidth, 0.0);
            OUTPUT(double, raw_bandwidth, 0.0);

            PARAM(std::vector<BoundingBox>, bounding_boxes, std::vector<BoundingBox>());
            PARAM(std::vector<Moment>, moments, std::vector<Moment>());
            PARAM(std::vector<float>, thresholds, std::vector<float>());
            PARAM(int, min_points, 100);
            PARAM(int, heartbeat_ms, 1000);
            PARAM(bool, force_active, false);
            
        MO_END;
        void  SetContext(mo::Context* ctx, bool overwrite = false);
        std::vector<mo::IParameter*> GetParameters(const std::string& filter) const;
        rcc::shared_ptr<EagleLib::Nodes::Node> h264_pass_through;
        mo::ITypedParameter<bool>* active_switch;
    protected:
        bool ProcessImpl();
    public:
        boost::posix_time::ptime last_keyframe_time;
        boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::rolling_mean>> raw_bandwidth_mean;
        boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::rolling_mean>> throttled_bandwidth_mean;
    };
}
#endif // HAVE_WT
