#pragma once
#include <Wt/WApplication>
#include <EagleLib/Nodes/Node.h>

namespace vclick
{
    class BoundingBox
    {
    public:
        inline bool Contains(const cv::Vec3f& point)
        {
            return point[0] > x && point[0] < x + width &&
                   point[1] > y && point[1] < y + height &&
                   point[2] > z && point[2] < z + depth;
        }
        inline bool Contains(const cv::Point3f& point)
        {
            return point.x > x && point.x < x + width &&
                   point.y > y && point.y < y + height &&
                   point.z > z && point.z < z + depth;
        }
        cv::Mat Contains(cv::Mat points);
        cv::Mat Contains(std::vector<cv::Vec3f>& points);
        template<typename AR> void serialize(AR& ar);

        float x,y,z;
        float width, height, depth;
    };

    class Moment
    {
    public:
        Moment(float Px_ = 0.0f, float Py_ = 0.0f, float Pz_ = 0.0f);
        template<typename AR> void serialize(AR& ar);
        float Evaluate(cv::Mat mask, cv::Mat points, cv::Vec3f centroid);
        float Px, Py, Pz;
    };

    class WebSink: public EagleLib::Nodes::Node
    {
    public:
        WebSink();
        MO_DERIVE(WebSink, EagleLib::Nodes::Node)
            INPUT(EagleLib::SyncedMemory, background_model, nullptr);
            INPUT(EagleLib::SyncedMemory, foreground_mask, nullptr);
            INPUT(EagleLib::SyncedMemory, point_cloud, nullptr);
            PARAM(std::vector<BoundingBox>, bounding_boxes, std::vector<BoundingBox>());
            PARAM(std::vector<Moment>, moments, std::vector<Moment>());
            PARAM(std::vector<float>, thresholds, std::vector<float>());
        MO_END;
        rcc::shared_ptr<EagleLib::Nodes::Node> h264_pass_through;
        mo::ITypedParameter<bool>* active_switch;
    protected:
        bool ProcessImpl();
    };

    class WebUi: public Wt::WApplication
    {
    public:
        // Blocking call, run on own thread.
        static void StartServer(int argc, char** argv, rcc::shared_ptr<WebSink> sink);
        WebUi(const Wt::WEnvironment& env);
        ~WebUi();
    private:
        void handleMove(int index, float x, float y, float z);
        void handleRotate(int index, float x, float y, float z);
        void handleResize(int index, float x, float y, float z);
        void handleActivate();
        void handleAddbb();
        void handleRebuildModel();
        void handleKeydown(int value);
        void handleBackgroundUpdate(mo::Context* ctx, mo::IParameter* param);
        void handleForegroundUpdate(mo::Context* ctx, mo::IParameter* param);
        
        Wt::WText* render_window;
        Wt::JSignal<int, float, float, float> onMove;
        Wt::JSignal<int, float, float, float> onResize;
        Wt::JSignal<int, float, float, float> onRotate;
        Wt::JSignal<int> onKeydown;
        Wt::JSlot* update;
        
        mo::TypedSlot<void(mo::Context*,mo::IParameter*)>* onBackgroundUpdate;
        mo::TypedSlot<void(mo::Context*, mo::IParameter*)>* onForegroundUpdate;
        mo::TypedInputParameterPtr<EagleLib::SyncedMemory> foreground_param;
        std::shared_ptr<mo::Connection> backgroundUpdateConnection;
        std::shared_ptr<mo::Connection> foregroundUpdateConnection;
        EagleLib::SyncedMemory* foreground;
    };
}