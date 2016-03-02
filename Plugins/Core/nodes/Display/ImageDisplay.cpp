#include "nodes/Display/ImageDisplay.h"
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <EagleLib/rcc/external_includes/cv_imgproc.hpp>
#include <parameters/UI/InterThread.hpp>
#include "../remotery/lib/Remotery.h"
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include <EagleLib/utilities/UiCallbackHandlers.h>
#include <EagleLib/utilities/WindowCallbackManager.h>
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/SystemTable.hpp"
#include "EagleLib/DataStreamManager.h"
#include <EagleLib/ParameteredObjectImpl.hpp>
#include "EagleLib/profiling.h"
using namespace EagleLib;
using namespace EagleLib::Nodes;

//NODE_DEFAULT_CONSTRUCTOR_IMPL(QtImageDisplay, Image, Sink, Display)
QtImageDisplay::QtImageDisplay():
    CpuSink()
{
}
static NodeInfo g_registerer_QtImageDisplay("QtImageDisplay", { "Image", "Sink", "Display"});
REGISTERCLASS(QtImageDisplay, &g_registerer_QtImageDisplay);
NODE_DEFAULT_CONSTRUCTOR_IMPL(KeyPointDisplay, Image, Sink, Display)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FlowVectorDisplay, Image, Sink, Display)
NODE_DEFAULT_CONSTRUCTOR_IMPL(HistogramDisplay, Image, Sink, Display)
NODE_DEFAULT_CONSTRUCTOR_IMPL(DetectionDisplay, Image, Sink, Display)
NODE_DEFAULT_CONSTRUCTOR_IMPL(OGLImageDisplay, Image, Sink, Display)



void QtImageDisplay::Init(bool firstInit)
{

}
TS<SyncedMemory> QtImageDisplay::doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    cv::Mat img = input.GetMat(stream);
    std::string display_name = getFullTreeName();
    EagleLib::cuda::scoped_event_stream_timer timer(stream, "QtImageDisplayTime");
	cuda::enqueue_callback_async(
		[this, display_name, img]()->void
	{
		rmt_ScopedCPUSample(QtImageDisplay_displayImage);
		PROFILE_FUNCTION;
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        auto manager = table->GetSingleton<WindowCallbackHandlerManager>();
        
        auto instance = manager->instance(GetDataStream()->get_stream_id());
        instance->imshow(display_name, img);
	}, stream);
    return input;
}
cv::cuda::GpuMat QtImageDisplay::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(img.channels() != 1 && img.channels() != 3)
    {
		NODE_LOG(warning) << "Image has " << img.channels() << " channels! Cannot display!";
        return img;
    }
	cv::Mat host_mat;
	std::string display_name;
	display_name = getFullTreeName();

    img.download(host_mat, stream);
    EagleLib::cuda::scoped_event_stream_timer timer(stream, "QtImageDisplayTime");
	cuda::enqueue_callback_async(
		[this, display_name, host_mat]()->void
	{
		rmt_ScopedCPUSample(QtImageDisplay_displayImage);
		PROFILE_FUNCTION;
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        auto manager = table->GetSingleton<WindowCallbackHandlerManager>();
        
        auto instance = manager->instance(GetDataStream()->get_stream_id());
        instance->imshow(display_name, host_mat);
	}, stream);
    
    return img;
}
void QtImageDisplay::doProcess(const cv::Mat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream)
{
	PROFILE_FUNCTION
    EagleLib::cuda::scoped_event_stream_timer timer(stream, "QtImageDisplayTime");
    cuda::enqueue_callback_async(
        [this, mat]()->void
    {
        rmt_ScopedCPUSample(QtImageDisplay_displayImage);
		PROFILE_FUNCTION;
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        auto manager = table->GetSingleton<WindowCallbackHandlerManager>();

        auto instance = manager->instance(GetDataStream()->get_stream_id());
        instance->imshow(getFullTreeName(), mat);
    }, stream);
}


void OGLImageDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
		updateParameter("Default Name", std::string("Default Name"))->SetTooltip("Set name for window");
    }
	prevName = *getParameter<std::string>(0)->Data();
	cv::namedWindow("Default Name", cv::WINDOW_OPENGL | cv::WINDOW_KEEPRATIO);
}



cv::cuda::GpuMat OGLImageDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(_parameters[0]->changed)
    {
        cv::destroyWindow(prevName);
        prevName = *getParameter<std::string>(0)->Data();
        _parameters[0]->changed = false;
        cv::namedWindow(prevName, cv::WINDOW_OPENGL | cv::WINDOW_KEEPRATIO);
    }
	std::string display_name = *getParameter<std::string>(0)->Data();
	cv::cuda::GpuMat display_buffer;
	img.copyTo(display_buffer, stream);
	cuda::enqueue_callback_async(
		[display_buffer, display_name]()->void
	{
		//cv::namedWindow(display_name, cv::WINDOW_OPENGL | cv::WINDOW_KEEPRATIO);
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        auto manager = table->GetSingleton<WindowCallbackHandlerManager>();
        auto instance = manager->instance(0);
        Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&WindowCallbackHandler::imshowd, instance, display_name, display_buffer, cv::WINDOW_OPENGL | cv::WINDOW_KEEPRATIO));
        //WindowCallbackHandler::instance()->imshow(display_name, display_buffer);
		//Parameters::UI::UiCallbackService::Instance()->post(
//			boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), 
					//display_name, display_buffer));

	}, stream);
    return img;
}

// This gets called in the user interface thread for drawing and displaying, after data is downloaded from the gpu
cv::Mat KeyPointDisplay::uicallback()
{
    return cv::Mat();
}

void KeyPointDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Device keypoints");
        addInputParameter<cv::Mat>("Host keypoints");
        updateParameter("Radius", int(5));
        updateParameter("Color", cv::Scalar(255,0,0));
        //hostData.resize(20);
        displayType = -1;
    }
}
void KeyPointDisplay::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    //SERIALIZE(hostData);
}
TS<SyncedMemory> KeyPointDisplay::doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* d_mat = getParameter<cv::cuda::GpuMat>(0)->Data();
	TIME
    if(d_mat && !d_mat->empty())
    {
		cv::Scalar color = *getParameter<cv::Scalar>(3)->Data();
		int radius = *getParameter<int>(2)->Data();
		std::string displayName = getFullTreeName();
		cv::Mat h_img, pts;
		TIME
		h_img = input.GetMat(stream);
        pts.create(1, 1000, CV_32FC2);
		d_mat->download(pts, stream);
		TIME
		EagleLib::cuda::enqueue_callback_async(
			[h_img, pts, radius, color, displayName]()->void
		{
			const cv::Vec2f* ptr = pts.ptr<cv::Vec2f>(0);
			for (int i = 0; i < pts.cols; ++i, ++ptr)
			{
				cv::circle(h_img, cv::Point(ptr->val[0], ptr->val[1]), radius, color, 1);
			}
			Parameters::UI::UiCallbackService::Instance()->post(
				boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), displayName, h_img));
		}, stream);
		TIME
        return input;
    }
    return input;
}
cv::cuda::GpuMat KeyPointDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* d_mat = getParameter<cv::cuda::GpuMat>(0)->Data();
	TIME
    if(d_mat && !d_mat->empty())
    {
		cv::Scalar color = *getParameter<cv::Scalar>(3)->Data();
		int radius = *getParameter<int>(2)->Data();
		std::string displayName = getFullTreeName();
		cv::Mat h_img, pts;
		TIME
		img.download(h_img, stream);
        pts.create(1, 1000, CV_32FC2);
		d_mat->download(pts, stream);
		TIME
		EagleLib::cuda::enqueue_callback_async(
			[h_img, pts, radius, color, displayName]()->void
		{
			const cv::Vec2f* ptr = pts.ptr<cv::Vec2f>(0);
			for (int i = 0; i < pts.cols; ++i, ++ptr)
			{
				cv::circle(h_img, cv::Point(ptr->val[0], ptr->val[1]), radius, color, 1);
			}
			Parameters::UI::UiCallbackService::Instance()->post(
				boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), displayName, h_img));
		}, stream);
		TIME
        return img;
    }
    auto h_mat = getParameter<cv::Mat>(1)->Data();
    if(h_mat)
    {

    }
    return img;
}

void FlowVectorDisplay::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
}

void FlowVectorDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Device initial poitns");
        addInputParameter<cv::cuda::GpuMat>("Device current points");
        addInputParameter<cv::cuda::GpuMat>("Point mask");
        updateParameter("Good Color", cv::Scalar(0,255,0));
        updateParameter("Bad Color", cv::Scalar(0,0,255));
    }
}

cv::cuda::GpuMat FlowVectorDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* d_initial = getParameter<cv::cuda::GpuMat>(0)->Data();
    cv::cuda::GpuMat* d_final = getParameter<cv::cuda::GpuMat>(1)->Data();
    cv::cuda::GpuMat* d_mask = getParameter<cv::cuda::GpuMat>(2)->Data();
    if(d_initial && !d_initial->empty() && d_final && !d_final->empty())
    {
		cv::Scalar goodColor = *getParameter<cv::Scalar>(3)->Data();
		cv::Scalar badColor = *getParameter<cv::Scalar>(4)->Data();
		cv::Mat h_img;
		img.download(h_img, stream);
		std::string displayName = getFullTreeName();
		if (d_mask)
		{
			cv::Mat h_initial, h_final, h_mask;
			d_initial->download(h_initial, stream);
			d_final->download(h_final, stream);
			d_mask->download(h_mask, stream);
			EagleLib::cuda::enqueue_callback_async(
				[h_img, h_initial, h_final, h_mask, goodColor, badColor, displayName]()->void
			{
				const cv::Vec2f* p1 = h_initial.ptr<cv::Vec2f>();
				const cv::Vec2f* p2 = h_final.ptr<cv::Vec2f>();
				const uchar* mask = h_mask.ptr<uchar>();
				for (int i = 0; i < h_initial.cols; ++i)
				{
					cv::line(h_img, cv::Point(p1[i].val[0], p1[i].val[1]), cv::Point(p2[i].val[0], p2[i].val[1]), mask[i] ? goodColor : badColor);
				}
				cv::imshow(displayName, h_img);
			}, stream);
		}
		else
		{
			cv::Mat h_initial, h_final;
			d_initial->download(h_initial, stream);
			d_final->download(h_final, stream);
			EagleLib::cuda::enqueue_callback_async(
				[h_img, h_initial, h_final, goodColor, displayName]()->void
			{
				const cv::Vec2f* p1 = h_initial.ptr<cv::Vec2f>();
				const cv::Vec2f* p2 = h_final.ptr<cv::Vec2f>();
				for (int i = 0; i < h_initial.cols; ++i)
				{
					cv::line(h_img, cv::Point(p1[i].val[0], p1[i].val[1]), cv::Point(p2[i].val[0], p2[i].val[1]), goodColor);
				}
				cv::imshow(displayName, h_img);
			}, stream);
		}

        //display(img, *d_initial, *d_final, d_mask ? *d_mask : cv::cuda::GpuMat(), fullTreeName, stream);
    }
    return img;
}

/*cv::Mat FlowVectorDisplay::uicallback()
{
    EventBuffer<cv::cuda::HostMem[4]>* buffer = hostData.waitBack();
    if(buffer)
    {
        cv::Mat img = buffer->data[0].createMatHeader();
        cv::Mat mask = buffer->data[1].createMatHeader();
        cv::Mat initial = buffer->data[2].createMatHeader();
        cv::Mat final = buffer->data[3].createMatHeader();
        cv::Scalar color = *getParameter<cv::Scalar>(3)->Data();
        // Iterate through all points and draw them vectors
        if(mask.empty())
        {
            for(int i = 0; i < final.cols; ++i)
            {
                cv::line(img, initial.at<cv::Point2f>(i), final.at<cv::Point2f>(i),color);
            }
        }
        return img;
    }
    return cv::Mat();
}*/
void histogramDisplayCallback(int status, void* userData)
{
    HistogramDisplay* node = (HistogramDisplay*)userData;
	Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&HistogramDisplay::displayHistogram, node));

}
void HistogramDisplay::displayHistogram()
{
    cv::cuda::HostMem* dataPtr = histograms.getBack();
    cv::Mat data = dataPtr->createMatHeader();
    if(data.channels() != 1)
    {
        //log(Error, "Currently only supports 1 channel histograms, input has " + boost::lexical_cast<std::string>(data.channels()) + " channels");
		NODE_LOG(error) << "Currently only supports 1 channel histograms, input has " << data.channels() << " channels";
        return;
    }
    double minVal, maxVal;
    int minIdx, maxIdx;
    cv::minMaxIdx(data, &minVal, &maxVal, &minIdx, &maxIdx);
    cv::Mat img(100, data.cols*5,CV_8U, cv::Scalar(0));
	updateParameter("Min value", minVal)->type =  Parameters::Parameter::State;
    updateParameter("Min bin", minIdx)->type = Parameters::Parameter::State;
	updateParameter("Max value", maxVal)->type = Parameters::Parameter::State;
	updateParameter("Max bin", maxIdx)->type =  Parameters::Parameter::State;
    for(int i = 0; i < data.cols; ++i)
    {
        double height = data.at<int>(i);
        height -= minVal;
        height /=(maxVal - minVal);
        height *= 100;
        cv::rectangle(img, cv::Rect(i*5, 100 - (int)height, 5, 100), cv::Scalar(255),-1);
    }
    cv::imshow(getFullTreeName(), img);

}

void HistogramDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input");
    }
}

cv::cuda::GpuMat HistogramDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat>(0)->Data();
    if(input)
    {
        if(input->rows == 1 || input->cols == 1)
        {
            input->download(*histograms.getFront(), stream);
            stream.enqueueHostCallback(histogramDisplayCallback,this);
            return img;
        }
    }
    if(img.rows == 1 || img.cols == 1)
    {
        img.download(*histograms.getFront(), stream);
        stream.enqueueHostCallback(histogramDisplayCallback,this);
    }
    // Currently assuming the input image is a histogram
    return img;
}
void DetectionDisplay_callback(int status, void* data)
{
    DetectionDisplay* node = static_cast<DetectionDisplay*>(data);
    node->displayCallback();
}

void DetectionDisplay::displayCallback()
{
    std::pair<cv::cuda::HostMem, std::vector<DetectedObject>>* data = hostData.getBack();
    cv::Mat h_img = data->first.createMatHeader();
    for(int i = 0; i < data->second.size(); ++i)
    {
        cv::rectangle(h_img, data->second[i].boundingBox, cv::Scalar(0,255,0), 1);
        for(int j = 0; j < data->second[i].detections.size(); ++j)
        {
            std::stringstream ss;
            ss << data->second[i].detections[j].classNumber << ":" << data->second[i].detections[j].confidence << " " << data->second[i].detections[j].label;
            cv::Point pos = data->second[i].boundingBox.tl() + cv::Point(0, -10);
            if(pos.x < 0)
                pos.x = 5;
            if(pos.x > h_img.cols - 100)
                pos.x = h_img.cols - 100;
            if(pos.y < 0)
                pos.y = 5;
            if(pos.y > h_img.rows - 100)
                pos.y = h_img.rows - 100;
        }
    }
	Parameters::UI::UiCallbackService::Instance()->post(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), getFullTreeName(), h_img));
}

void DetectionDisplay::Init(bool firstInit)
{
    addInputParameter<DetectedObject>("Input detections");
}

cv::cuda::GpuMat DetectionDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    auto buf = hostData.getFront();
    img.download(buf->first, stream);

    stream.enqueueHostCallback(DetectionDisplay_callback, this);
    return img;
}
