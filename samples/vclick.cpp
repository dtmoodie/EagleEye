//#include "vclick.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "vclick.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"

#include <cereal/types/vector.hpp>

#include "EagleLib/IO/memory.hpp"
#include "EagleLib/Nodes/NodeFactory.h"
#include "EagleLib/Nodes/NodeInfo.hpp"

#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>
#include <Wt/WJavaScriptSlot>
#include <Wt/WHBoxLayout>
#include <Wt/WVBoxLayout>

#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include <boost/bind/placeholders.hpp>

#include <fstream>

using namespace EagleLib;
using namespace EagleLib::Nodes;
using namespace vclick;
using namespace Wt;
//using namespace boost::placeholders;
INSTANTIATE_META_PARAMETER(BoundingBox);
INSTANTIATE_META_PARAMETER(std::vector<BoundingBox>);
INSTANTIATE_META_PARAMETER(Moment);
INSTANTIATE_META_PARAMETER(std::vector<Moment>);

cv::Mat BoundingBox::Contains(std::vector<cv::Vec3f>& points)
{
    return Contains(cv::Mat(1, points.size(), CV_32FC3, &points[0]));
}

cv::Mat BoundingBox::Contains(cv::Mat points)
{
    cv::Mat output_mask;
    output_mask.create(points.size(), CV_8UC1);
    const int num_points = points.size().area();
    uchar* mask_ptr = output_mask.ptr<uchar>();
    cv::Vec3f* pt_ptr = points.ptr<cv::Vec3f>();
    for(int i = 0; i < num_points; ++i)
    {
        const cv::Vec3f& pt = pt_ptr[i];
        mask_ptr[i] = Contains(pt);
    }
    return output_mask;
}

template<typename AR> 
void BoundingBox::serialize(AR& ar)
{
    ar(CEREAL_NVP(x));
    ar(CEREAL_NVP(y));
    ar(CEREAL_NVP(z));
    ar(CEREAL_NVP(width));
    ar(CEREAL_NVP(height));
    ar(CEREAL_NVP(depth));
}

Moment::Moment(float Px_, float Py_, float Pz_):
    Px(Px_), Py(Py_), Pz(Pz_)
{

}

template<typename AR> 
void Moment::serialize(AR& ar)
{
    ar(CEREAL_NVP(Px));
    ar(CEREAL_NVP(Py));
    ar(CEREAL_NVP(Pz));
}

float Moment::Evaluate(cv::Mat mask, cv::Mat points, cv::Vec3f centroid)
{
    float value = 0;
    uchar* mask_ptr = mask.ptr<uchar>();
    cv::Vec3f* pts = points.ptr<cv::Vec3f>();
    const int num_points = mask.size().area();
    float count = 0;
    for(int i = 0; i < num_points; ++i)
    {
        if(mask_ptr[i])
        {
            value += pow(pts[i][0] - centroid[0], Px) * pow(pts[i][1] - centroid[1], Py) * pow(pts[i][2] - centroid[2], Pz); 
            ++count;
        }
    }
    value /= count;
    return value;
}

WebSink::WebSink()
{
    h264_pass_through = mo::MetaObjectFactory::Instance()->Create("h264_pass_through");
    active_switch = h264_pass_through->GetParameter<bool>("active");
    moments.emplace_back(2, 0, 0);
    moments.emplace_back(2, 2, 0);
    moments.emplace_back(2, 0, 2);
}

bool WebSink::ProcessImpl()
{
    std::vector<cv::Vec3f> foreground_points;
    cv::Mat mask = foreground_mask->GetMat(Stream());
    cv::Mat ptCloud = point_cloud->GetMat(Stream());
    int points = cv::countNonZero(mask);
    foreground_points.reserve(points);
    for(int i = 0; i < ptCloud.rows; ++i)
    {
        for(int j = 0; j < ptCloud.cols; ++j)
        {
            if(mask.at<uchar>(i,j))
            {
                foreground_points.push_back(ptCloud.at<cv::Vec3f>(i,j));
            }
        }
    }
    for(auto& bb : bounding_boxes)
    {
        cv::Mat bb_mask = bb.Contains(foreground_points);
        cv::Vec3f centroid(0,0,0);
        uchar* mask_ptr = bb_mask.ptr<uchar>();
        float count = 0;
        for(int i = 0; i < foreground_points.size(); ++i)
        {
            if(mask_ptr[i])
            {
                centroid += foreground_points[i];
                ++count;
            }
        }
        if(count == 0.0)
            continue;
        centroid /= count;
        for(int i = 0; i < moments.size(); ++i)
        {
            float value = moments[i].Evaluate(bb_mask, 
                cv::Mat(1, foreground_points.size(), CV_32FC3, &foreground_points[0]), centroid);
            if(value > thresholds[i])
            {
                active_switch->UpdateData(true, point_cloud_param.GetTimestamp(), _ctx);
            }
        }
    }
    return false;
}

rcc::shared_ptr<WebSink> g_sink;

WApplication* createApplication(const Wt::WEnvironment& env)
{
    return new WebUi(env);
}

void WebUi::StartServer(int argc, char** argv, rcc::shared_ptr<WebSink> sink)
{
    g_sink = sink;
    Wt::WRun(argc, argv, &createApplication);
}

WebUi::WebUi(const Wt::WEnvironment& env):
    WApplication(env),
    onMove(this, "onMove"),
    onResize(this, "onResize"),
    onRotate(this, "onRotate"),
    onKeydown(this, "onKeydown")
{
    setTitle("vclick3d demo");
    
    onBackgroundUpdate = new mo::TypedSlot<void(mo::Context*, mo::IParameter*)>(
        std::bind(&WebUi::handleBackgroundUpdate, this, std::placeholders::_1, std::placeholders::_2));

    onForegroundUpdate = new mo::TypedSlot<void(mo::Context*, mo::IParameter*)>(
        std::bind(&WebUi::handleForegroundUpdate, this, std::placeholders::_1, std::placeholders::_2));

    auto stream = g_sink->GetDataStream();
    auto fg = stream->GetNode("ForegroundEstimate0");
    foreground_param.SetUserDataPtr(&foreground);
    foreground_param.SetName("foreground");
    backgroundUpdateConnection = fg->GetParameter("background_model")->RegisterUpdateNotifier(onBackgroundUpdate);
    auto fg_param = fg->GetParameter("foreground");
    fg_param->Subscribe();
    foreground_param.SetInput(fg_param);
    foregroundUpdateConnection = foreground_param.RegisterUpdateNotifier(onForegroundUpdate);

    
    {
        std::ofstream ofs("./web/bb.json");
        cereal::JSONOutputArchive ar(ofs);
        auto bb = g_sink->GetParameter<std::vector<BoundingBox>>("bounding_boxes");

        auto func = mo::SerializationFunctionRegistry::Instance()->
            GetJsonSerializationFunction(bb->GetTypeInfo());
        if (func)
        {
            func(bb, ar);
        }
    }

    Wt::WPushButton *activate = new WPushButton("Activate", root());
    activate->clicked().connect(this, &WebUi::handleActivate);
    

    Wt::WPushButton *trigger = new WPushButton("AddTriggerRegion", root());
    trigger->clicked().connect(this, &WebUi::handleAddbb);
    

    Wt::WPushButton *rebuild = new WPushButton("Rebuild Model", root());
    rebuild->clicked().connect(this, &WebUi::handleRebuildModel);
    

    if (!require("three.js"))
    {
        BOOST_LOG_TRIVIAL(info) << "Unable to open three.js";
    }

    if (!require("js/libs/stats.min.js"))
    {
        BOOST_LOG_TRIVIAL(info) << "Unable to open stats.min.js";
    }
    if(!require("js/controls/TrackballControls.js"))
    {
        BOOST_LOG_TRIVIAL(info) << "Unable to open TrackballControls.js";
    }
    if(!require("js/controls/OrbitControls.js"))
    {
        BOOST_LOG_TRIVIAL(error) << "Failed to load OrbitControls.js";
    }
    if(!require("js/controls/TransformControls.js"))
    {
        BOOST_LOG_TRIVIAL(error) << "Failed to load TransformControls.js";
    }
    if(!require("js/controls/DragControls.js"))
    {
        BOOST_LOG_TRIVIAL(error) << "Failed to load DragControls.js";
    }

    onMove.connect(boost::bind(&WebUi::handleMove, this, _1, _2, _3, _4));
    onKeydown.connect(boost::bind(&WebUi::handleKeydown, this, _1));

    std::ifstream ifs;
    ifs.open("../render_script.txt");
    if(ifs.is_open())
    {
        std::string move_call = onMove.createCall("SELECTED.index", "SELECTED.position.x", 
            "SELECTED.position.y", "SELECTED.position.z");
        std::string source((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        std::stringstream ss;
        ss << source;
        ss << "\n";
        ss << "function onDocumentMouseUp( event ) {\n";
        ss << "  event.preventDefault();\n";
        ss << "  controls.enabled = true;\n";
        ss << "  if (INTERSECTED) {\n";
        ss << move_call;
        ss << "      SELECTED = null;\n";
        ss << "  }\n";
        ss << "  container.style.cursor = 'auto';\n";
        ss << "}\n";
        ss << "function onDocumentKeydown(event){\n";
        ss << "  var key = event.which; \n";
        ss <<    onKeydown.createCall("key");
        ss << "  if(key == 65){\n";
        ss << "    addBoundingBox();\n";
        ss << "  }\n";
        ss << "}\n";
        render_window = new Wt::WText(root());
        render_window->doJavaScript(ss.str());
    }
    
}

WebUi::~WebUi()
{
    delete onForegroundUpdate;
    delete onBackgroundUpdate;
}
void WebUi::handleMove(int index, float x, float y, float z)
{
    if(index >= 0 && index < g_sink->bounding_boxes.size())
    {
        g_sink->bounding_boxes[index].x = x;
        g_sink->bounding_boxes[index].y = y;
        g_sink->bounding_boxes[index].z = z;
    }
}

void WebUi::handleRotate(int index, float x, float y, float z)
{

}

void WebUi::handleResize(int index, float x, float y, float z)
{

}

void WebUi::handleActivate()
{
    g_sink->active_switch->UpdateData(!g_sink->active_switch->GetData());
}

void WebUi::handleAddbb()
{
    BoundingBox bb;
    bb.x = 0;
    bb.y = 0;
    bb.z = 0;
    bb.width = 20;
    bb.height = 30;
    bb.depth = 20;
    g_sink->bounding_boxes.push_back(bb);
}

void WebUi::handleRebuildModel()
{
    auto stream = g_sink->GetDataStream();
    auto node = stream->GetNode("ForegroundEstimate0");
    if(node)
    {
        auto param = node->GetParameter<bool>("build_model");
        if(param)
        {
            param->UpdateData(true);
        }
    }
}

void WebUi::handleKeydown(int value)
{
    if(value == 32)
    {
        handleRebuildModel();
    }else if(value == 65)
    {
        handleAddbb();
    }else if(value == 187)
    {
        handleActivate();
    }

    std::cout << value << std::endl;
}

void WebUi::handleBackgroundUpdate(mo::Context* ctx, mo::IParameter* param)
{
    auto func = mo::SerializationFunctionRegistry::Instance()->
        GetJsonSerializationFunction(param->GetTypeInfo());
    dynamic_cast<mo::ITypedParameter<SyncedMemory>*>(param)->GetDataPtr()->Synchronize();
    if(func)
    {
        std::ofstream ofs("./web/background.json");
        cereal::JSONOutputArchive ar(ofs);
        func(param, ar);
    }
}

void WebUi::handleForegroundUpdate(mo::Context* ctx, mo::IParameter* param)
{
    auto func = mo::SerializationFunctionRegistry::Instance()->
        GetJsonSerializationFunction(param->GetTypeInfo());
    dynamic_cast<mo::ITypedParameter<SyncedMemory>*>(param)->GetDataPtr()->Synchronize();
    if (func)
    {
        std::ofstream ofs("./web/foreground.json");
        cereal::JSONOutputArchive ar(ofs);
        func(param, ar);
    }
}
MO_REGISTER_CLASS(WebSink);
