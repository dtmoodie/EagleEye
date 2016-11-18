#ifdef HAVE_WT
#pragma once
#include <EagleLib/Nodes/Node.h>
#include "BoundingBox.hpp"
#include "Moment.hpp"

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
            OUTPUT(cv::Mat, foreground_points, cv::Mat());
            PARAM(std::vector<BoundingBox>, bounding_boxes, std::vector<BoundingBox>());
            PARAM(std::vector<Moment>, moments, std::vector<Moment>());
            PARAM(std::vector<float>, thresholds, std::vector<float>());
            PARAM(int, min_points, 100);
        MO_END;
        void  SetContext(mo::Context* ctx, bool overwrite = false);
        std::vector<mo::IParameter*> GetParameters(const std::string& filter) const;
        rcc::shared_ptr<EagleLib::Nodes::Node> h264_pass_through;
        mo::ITypedParameter<bool>* active_switch;
    protected:
        bool ProcessImpl();
    };
}
#endif // HAVE_WT
