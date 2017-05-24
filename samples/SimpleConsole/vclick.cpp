#ifdef HAVE_WT

#include "WebSink.hpp"
#include "vclick.hpp"

#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/serialization/SerializationFactory.hpp"

#include <cereal/types/vector.hpp>
#include <Aquila/IO/cvMat.hpp>
#include "Aquila/IO/memory.hpp"
#include "Aquila/nodes/NodeFactory.hpp"


#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>
#include <Wt/WJavaScriptSlot>
#include <Wt/WHBoxLayout>
#include <Wt/WVBoxLayout>
#include <Wt/WPanel>
#include <Wt/WImage>
#include <Wt/WVideo>
#include <Wt/WEnvironment>
#include <Wt/WServer>
#include <Wt/WStandardItemModel>
#include <Wt/WBatchEditProxyModel>
#include <Wt/WStandardItem>
#include "Wt/Chart/WCartesianChart"

#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include <boost/bind/placeholders.hpp>


#include <fstream>

using namespace aq;
using namespace aq::Nodes;
using namespace vclick;
using namespace Wt;

rcc::shared_ptr<WebSink> g_sink;
/*class DelayedBatchEditProxyModel: public Wt::WBatchEditProxyModel
{
public:
    bool setData(const WModelIndex& index,
        const boost::any& value, int role)
    {
        Wt::Item *item = itemFromIndex(index.parent());

        ValueMap::iterator i
            = item->editedValues_.find(Wt::Cell(index.row(), index.column()));

        if (i == item->editedValues_.end()) {
            WModelIndex sourceIndex = mapToSource(index);
            DataMap dataMap;

            if (sourceIndex.isValid())
                dataMap = sourceModel()->itemData(sourceIndex);

            dataMap[role] = value;

            if (role == EditRole)
                dataMap[DisplayRole] = value;

            item->editedValues_[Wt::Cell(index.row(), index.column())] = dataMap;
        }
        else {
            i->second[role] = value;
            if (role == EditRole)
                i->second[DisplayRole] = value;
        }

        //dataChanged().emit(index, index);

        return true;
    }
    void commitAll()
    {
        Wt::WBatchEditProxyModel::commitAll();
        auto tl = index(0,0);

        dataChanged().emit(index, index);
    }
};*/


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
    Wt::WContainerWidget *container = new Wt::WContainerWidget(root());
    setTitle("vclick3d demo");
    Wt::WLogger& logger = env.server()->logger();
    logger.configure("* -debug -info");
    auto stream = g_sink->getDataStream();
    auto fg = stream->getNode("ForegroundEstimate0");

    auto bg_param = fg->GetParameter("background_model");
    backgroundStream.reset(new TParameterResource<aq::SyncedMemory>(this,
        fg->GetParameter("background_model"), "background_model"));
    backgroundStream->handleParamUpdate(nullptr, nullptr);

    foregroundStream.reset(new TParameterResource<cv::Mat>(this,
        g_sink->GetParameter("foreground_points"), "foreground"));
    foregroundStream->handleParamUpdate(nullptr, nullptr);

    boundingBoxStream.reset(new TParameterResource<std::vector<BoundingBox>>(this,
        g_sink->GetParameter("bounding_boxes"), "bounding_boxes"));
    boundingBoxStream->handleParamUpdate(nullptr, nullptr);


    
        bandwidth_raw.set_capacity(500);
        bandwidth_throttled.set_capacity(500);
        timestamp.set_capacity(500);
        auto tb = g_sink->GetParameter<double>("throttled_bandwidth");
        auto rb = g_sink->GetParameter<double>("raw_bandwidth");

        onBandwidthRawUpdateSlot.reset(new mo::ParameterUpdateSlot(std::bind(
            [this, rb](mo::Context* ctx, mo::IParam* param)
        {
            this->bandwidthRawUpdated = true;
            this->bandwidth_raw.push_back(rb->GetData());
            this->updatePlots();
        }, std::placeholders::_1, std::placeholders::_2)));
        onRawBandwidthUpdateConnection = rb->registerUpdateNotifier(onBandwidthRawUpdateSlot.get());

        onBandwidthThrottledUpdateSlot.reset(new mo::ParameterUpdateSlot(std::bind(
            [this, tb](mo::Context* ctx, mo::IParam* param)
        {
            this->bandwidthThrottledUpdated = true;
            this->bandwidth_throttled.push_back(tb->GetData());
            this->updatePlots();
        }, std::placeholders::_1, std::placeholders::_2)));
        onThrottledBandwidthUpdateConnection = tb->registerUpdateNotifier(onBandwidthThrottledUpdateSlot.get());
    if (false)
    {
        Wt::WAnimation animation(Wt::WAnimation::SlideInFromTop,
            Wt::WAnimation::EaseOut,
            100);
        model_proxy = new WBatchEditProxyModel();
        model = new WStandardItemModel(100, 3, this);
        model_proxy->setSourceModel(model);
        model->setHeaderData(0, WString("Time"));
        model->setHeaderData(1, WString("Raw Bandwidth"));
        model->setHeaderData(2, WString("VClick3d Bandwidth"));



        Wt::WPanel *plot_panel = new Wt::WPanel(container);
        plot_panel->setTitle("Bandwidth usage");
        plot_panel->addStyleClass("centered-example");
        plot_panel->setCollapsible(true);
        plot_panel->setAnimation(animation);


        chart = new Wt::Chart::WCartesianChart(container);
        chart->setModel(model);        // set the model
        chart->setXSeriesColumn(0);    // set the column that holds the X data
        chart->setLegendEnabled(true); // enable the legend
        chart->setZoomEnabled(true);
        chart->setPanEnabled(true);
        chart->setCrosshairEnabled(true);
        chart->setBackground(WColor(200, 200, 200));
        chart->setType(Wt::Chart::ScatterPlot);   // set type to ScatterPlot
        chart->setAutoLayoutEnabled();
        chart->resize(800, 300);
        for (int i = 1; i < 3; ++i) {
            Wt::Chart::WDataSeries *s = new Wt::Chart::WDataSeries(i, Wt::Chart::LineSeries);
            s->setShadow(WShadow(3, 3, WColor(0, 0, 0, 127), 3));
            chart->addSeries(s);
        }

        plot_panel->setCentralWidget(chart);
        chart->setMargin(10, Top | Bottom);            // add margin vertically
        chart->setMargin(WLength::Auto, Left | Right); // center horizontally

        
    }
    

    /*heartbeatStream = new TParameterResourceRaw<cv::Mat>(this,
        g_sink->GetParameter("output_jpeg"), "heartbeat");
    heartbeatStream->handleParamUpdate(nullptr, nullptr);*/

    /*auto jpeg_node = stream->getNode("JPEGSink0");
    rawStream = nullptr;
    if(jpeg_node)
    {
        rawStream = new TParameterResourceRaw<cv::Mat>(this,
            jpeg_node->GetParameter("jpeg_buffer"), "rawvideo");
        rawStream->handleParamUpdate(nullptr, nullptr);
    }*/
    

    

    /*
    {
        Wt::WPanel *raw_video_panel = new Wt::WPanel(container);
            raw_video_panel->setTitle("Raw video");
            raw_video_panel->addStyleClass("centered-example");
            raw_video_panel->setCollapsible(true);
            raw_video_panel->setAnimation(animation);

        Wt::WVideo* raw_feed = new Wt::WVideo(container);
            raw_video_panel->setCentralWidget(raw_feed);
            raw_feed->addSource(Wt::WLink("http://192.168.1.252:8090"));
            raw_video_panel->collapsed().connect(std::bind([=]()
            {
                raw_feed->pause();
            }));
            
    }
    {
        Wt::WPanel *video_panel = new Wt::WPanel(container);
            video_panel->setTitle("Heartbeat video");
            video_panel->addStyleClass("centered-example");
            video_panel->setCollapsible(true);
            video_panel->setAnimation(animation);

        Wt::WVideo* video_feed = new Wt::WVideo(container);
            video_panel->setCentralWidget(video_feed);
            video_feed->addSource(Wt::WLink("http://192.168.1.252:8080"));
            video_panel->collapsed().connect(std::bind([=]()
            {
                video_feed->pause();
            }));
            onActivate.reset(new mo::TSlot<void(mo::Context*, mo::IParam*)>(std::bind(
                [=](mo::Context* ctx, mo::IParam* param)
            {
                if(param->GetData<bool>())
                {
                    auto lock = this->getUpdateLock();
                    video_feed->play();
                }
            }, std::placeholders::_1, std::placeholders::_2)));
            onActivateConntection = g_sink->active_switch->registerUpdateNotifier(onActivate.get());
    }*/

    auto background_link = Wt::WLink(backgroundStream.get());
    auto foreground_link = Wt::WLink(foregroundStream.get());
    auto boundingBox_link = Wt::WLink(boundingBoxStream.get());

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
        ss << "var bounding_box_url = \"" << boundingBox_link.url() << "\";\n";
        ss << "var background_url = \"" << background_link.url() << "\";\n";
        ss << "var foregorund_url = \"" << foreground_link.url() << "\";\n";
        ss << source;
        ss << "\n" << 
            "function onDocumentKeydown(event) {"
            "    var key = event.which;"
            <<   onKeydown.createCall("key") <<
            "    if (key == 65) {"
            "        addBoundingBox();"
            "    }"
            "    else if (key == 82) {"
            "        loadBb();"
            "        loadForeground();"
            "        loadBackground();"
            "    }"
            "}"
            "function onDocumentMouseUp( event ) {\n"
            "    event.preventDefault();\n"
            "    controls.enabled = true;\n"
            "    if (INTERSECTED) {\n        "
            <<       move_call <<
            "        SELECTED = null;\n"
            "    }\n"
            "    container.style.cursor = 'auto';\n"
            "}\n";

        std::string js = ss.str();
        LOG(debug) << "\n" << js;

        render_window = new Wt::WText(container);
        
        render_window->doJavaScript(js);
    }else
    {
        LOG(error) <<"Unable to open ../render_script.txt";
    }
    
}

WebUi::~WebUi()
{
    
}

void WebUi::handleMove(int index, float x, float y, float z)
{
    if(index >= 0 && index < g_sink->bounding_boxes.size())
    {
        g_sink->bounding_boxes[index].x = x;
        g_sink->bounding_boxes[index].y = y;
        g_sink->bounding_boxes[index].z = z;
        g_sink->bounding_boxes_param.emitUpdate();
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
    g_sink->active_switch->updateData(!g_sink->active_switch->GetData());
}

void WebUi::handleAddbb()
{
    BoundingBox bb;
    bb.x = 0;
    bb.y = 0;
    bb.z = 0;
    bb.width = 800;
    bb.height = 1200;
    bb.depth = 800;
    g_sink->bounding_boxes.push_back(bb);
    g_sink->bounding_boxes_param.emitUpdate();
}

void WebUi::handleRebuildModel()
{
    auto stream = g_sink->getDataStream();
    auto node = stream->getNode("ForegroundEstimate0");
    if(node)
    {
        auto param = node->GetParameter<bool>("build_model");
        if(param)
        {
            param->updateData(true);
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
void WebUi::updatePlots()
{
    if(bandwidthRawUpdated && bandwidthThrottledUpdated)
    {
        timestamp.push_back(++current_framenumber);
        /*if(int(current_timestamp) % update_frequency == 0)
        {
            size_t num_samples = std::min(std::min(timestamp.size(), bandwidth_raw.size()), bandwidth_throttled.size());
            if (num_samples >= 99)
            {
                int model_index = 99;
                auto applock = this->getUpdateLock();
                double maxBw = 0;
                for (int i = num_samples - 1; i >= 0 && model_index >= 0; --i, --model_index)
                {
                    model_proxy->setData(model_proxy->index(model_index, 0), i);
                    model_proxy->setData(model_proxy->index(model_index, 1), bandwidth_raw[i]);
                    model_proxy->setData(model_proxy->index(model_index, 2), bandwidth_throttled[i]);
                    maxBw = std::max(bandwidth_raw[i], maxBw);
                    maxBw = std::max(bandwidth_throttled[i], maxBw);
                }
                model_proxy->commitAll();

                Wt::Chart::WAxis& x_axis = chart->axis(Wt::Chart::Axis::XAxis);
                x_axis.setRange(timestamp[0], timestamp[99]);
                Wt::Chart::WAxis& y1_axis = chart->axis(Wt::Chart::Axis::Y1Axis);
                Wt::Chart::WAxis& y2_axis = chart->axis(Wt::Chart::Axis::Y2Axis);
                
                y1_axis.setRange(0, maxBw);
                y1_axis.setGridLinesEnabled(true);
                y1_axis.setLabelInterval(maxBw/10);

                y2_axis.setRange(0, maxBw);
                y2_axis.setVisible(true);
                y2_axis.setLabelInterval(maxBw/10);
            }
        }*/
        if(bandwidth_raw.size() == bandwidth_raw.capacity() &&
            bandwidth_throttled.size() == bandwidth_throttled.capacity() &&
            int(current_framenumber) % update_frequency == 0)
        {
            cv::Mat plot(500, 500, CV_8UC3);
            plot.setTo(cv::Scalar(0, 0, 0));
            // Find max
            double maxValue = 0;
            for(const auto& itr : bandwidth_raw)
            {
                maxValue = std::max(itr, maxValue);
            }
            maxValue *= 1.05;
            int x = 0;
            for(const auto& itr : bandwidth_raw)
            {
                double value = std::max<double>(0, std::min<double>(500 - itr * 500.0 / maxValue, 499));
                plot.at<cv::Vec3b>(value, x) = cv::Vec3b(0,0,255);
                ++x;
            }
            x = 0;
            for (const auto& itr : bandwidth_throttled)
            {
                double value = std::max<double>(0, std::min<double>(500 - itr * 500.0 / maxValue, 499));
                plot.at<cv::Vec3b>(value, x) = cv::Vec3b(0, 255, 0);
                ++x;
            }
            mo::ThreadSpecificQueue::push(
                [plot]()
                {
                    cv::imshow("bandwidth plot", plot);
                },mo::ThreadRegistry::instance()->getThread(mo::ThreadRegistry::GUI));
        }
        
        
        bandwidthRawUpdated = false;
        bandwidthThrottledUpdated = false;
    }
}

#endif
