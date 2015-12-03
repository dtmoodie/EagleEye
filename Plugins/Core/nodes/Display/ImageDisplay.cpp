#include "nodes/Display/ImageDisplay.h"
#include <external_includes/cv_core.hpp>
#include <external_includes/cv_imgproc.hpp>
#include <UI/InterThread.hpp>
#include "../remotery/lib/Remotery.h"
#include <EagleLib/utilities/CudaCallbacks.hpp>
using namespace EagleLib;

NODE_DEFAULT_CONSTRUCTOR_IMPL(QtImageDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(OGLImageDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(KeyPointDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FlowVectorDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(HistogramDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(DetectionDisplay)

REGISTER_NODE_HIERARCHY(QtImageDisplay, Image, Sink, Display)
REGISTER_NODE_HIERARCHY(OGLImageDisplay, Image, Sink, Display)
REGISTER_NODE_HIERARCHY(KeyPointDisplay, Image, Sink, Display)
REGISTER_NODE_HIERARCHY(FlowVectorDisplay, Image, Sink, Display)
REGISTER_NODE_HIERARCHY(HistogramDisplay, Image, Sink, Display)
REGISTER_NODE_HIERARCHY(DetectionDisplay, Image, Sink, Display)

QtImageDisplay::QtImageDisplay(boost::function<void(cv::Mat, Node*)> cpuCallback_)
{
}
QtImageDisplay::QtImageDisplay(boost::function<void (cv::cuda::GpuMat, Node*)> gpuCallback_)
{
}
void QtImageDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
		updateParameter("Name", std::string(), Parameters::Parameter::Control, "Set name for window");
    }
}
struct UserData
{
    UserData(cv::cuda::HostMem img, QtImageDisplay* node_): displayImage(img), node(node_){}
    cv::cuda::HostMem displayImage;
    QtImageDisplay* node;
};

void QtImageDisplay_cpuCallback(int status, void* userData)
{
    UserData* tmp = (UserData*)userData;
	Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&QtImageDisplay::displayImage, tmp->node, tmp->displayImage));
    delete tmp;
}

void QtImageDisplay::displayImage(cv::cuda::HostMem image)
{
	rmt_ScopedCPUSample(QtImageDisplay_displayImage);
    std::string name = *getParameter<std::string>(0)->Data();
    if(name.size() == 0)
    {
        name = fullTreeName;
    }
    try
    {
        cv::imshow(name, image.createMatHeader());
    }catch(cv::Exception &err)
    {
		NODE_LOG(warning) << err.what();
    }
    parameters[0]->changed = false;
}

cv::cuda::GpuMat
QtImageDisplay::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(img.channels() != 1 && img.channels() != 3)
    {
		NODE_LOG(warning) << "Image has " << img.channels() << " channels! Cannot display!";
        return img;
    }

    img.download(hostImage, stream);
    stream.enqueueHostCallback(QtImageDisplay_cpuCallback, new UserData(hostImage,this));
    return img;
}



void OGLImageDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
		updateParameter("Default Name", std::string("Default Name"), Parameters::Parameter::Control, "Set name for window");
    }
	prevName = *getParameter<std::string>(0)->Data();
	cv::namedWindow("Default Name", cv::WINDOW_OPENGL);
}
struct oglData
{
	std::string name;
	Buffer<cv::cuda::GpuMat, EventPolicy>* data;
};
void oglDisplay(std::string name, cv::cuda::GpuMat data)
{
	cv::namedWindow(name, cv::WINDOW_OPENGL);
	cv::imshow(name, data);
}
void oglCallback(int status, void* user_data)
{
	oglData* data = static_cast<oglData*>(user_data);
	Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&oglDisplay,data->name, data->data->data));
	delete data;
}

void OGLImageDisplay::display()
{
	cv::namedWindow(prevName, cv::WINDOW_OPENGL);
	auto buffer = bufferPool.getFront();
	cv::imshow(prevName, buffer->data);
}

cv::cuda::GpuMat OGLImageDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed)
    {
        cv::destroyWindow(prevName);
        prevName = *getParameter<std::string>(0)->Data();
        parameters[0]->changed = false;
        cv::namedWindow(prevName, cv::WINDOW_OPENGL);
    }
	auto buffer = bufferPool.getFront();
	auto userData = new oglData;
	userData->data = buffer;
	userData->name = prevName;
	img.copyTo(userData->data->data, stream);
	stream.enqueueHostCallback(oglCallback, userData);
    return img;
}

// This gets called in the user interface thread for drawing and displaying, after data is downloaded from the gpu
cv::Mat KeyPointDisplay::uicallback()
{
    /*if(displayType == 0)
    {
        EventBuffer<std::pair<cv::cuda::HostMem,cv::cuda::HostMem>>* buffer = hostData.waitBack();
        cv::Scalar color = *getParameter<cv::Scalar>(3)->Data();
        int radius = *getParameter<int>(2)->Data();
        if(buffer)
        {
            cv::Mat keyPoints = buffer->data.first.createMatHeader();
            cv::Mat hostImage = buffer->data.second.createMatHeader();
            cv::Vec2f* pts = keyPoints.ptr<cv::Vec2f>(0);
            for(int i = 0; i < keyPoints.cols; ++i, ++pts)
            {
                cv::circle(hostImage, cv::Point(pts->val[0], pts->val[1]), radius, color, 1);
            }
            return hostImage;
        }
        return cv::Mat();
    }*/
    return cv::Mat();
}

/*
void KeyPointDisplay_callback(int status, void* userData)
{
    KeyPointDisplay* node = (KeyPointDisplay*)userData;
    cv::Mat img = node->uicallback();
    try
    {
		if (!img.empty())
		{
			Parameters::UI::UiCallbackService::Instance()->post(
				boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), node->fullTreeName, img));
		}
            
    }catch(cv::Exception &e)
    {
        std::cout << e.what() << std::endl;
    }

}*/

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
cv::cuda::GpuMat KeyPointDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* d_mat = getParameter<cv::cuda::GpuMat>(0)->Data();
	TIME
    if(d_mat && !d_mat->empty())
    {
		cv::Scalar color = *getParameter<cv::Scalar>(3)->Data();
		int radius = *getParameter<int>(2)->Data();
		std::string displayName = fullTreeName;
		cv::Mat h_img, pts;
		TIME
		img.download(h_img, stream);
		d_mat->download(pts, stream);
		TIME
		EagleLib::cuda::enqueue_callback_async(
			[h_img, pts, radius, color, displayName]()->void
		{
			//cv::Mat pts = h_points.createMatHeader();
			const cv::Vec2f* ptr = pts.ptr<cv::Vec2f>(0);
			for (int i = 0; i < pts.cols; ++i, ++ptr)
			{
				cv::circle(h_img, cv::Point(ptr->val[0], ptr->val[1]), radius, color, 1);
			}
			Parameters::UI::UiCallbackService::Instance()->post(
				boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), displayName, h_img));
		}, stream);
		TIME
        //auto buffer = hostData.getFront();
        //d_mat->download(buffer->data.first, stream);
        //img.download(buffer->data.second, stream);
        //buffer->fillEvent.record(stream);
        //stream.enqueueHostCallback(KeyPointDisplay_callback, this);
        //displayType = 0;
        return img;
    }
    auto h_mat = getParameter<cv::Mat>(1)->Data();
    if(h_mat)
    {

    }
    return img;
}
/*void FlowVectorDisplay_callback(int status, void* userData)
{
    FlowVectorDisplay* node = (FlowVectorDisplay*)userData;
	Parameters::UI::UiCallbackService::Instance()->post(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), node->displayName, node->uicallback()));
}*/

void FlowVectorDisplay::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    //SERIALIZE(hostData);
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
/*void FlowVectorDisplay::display(cv::cuda::GpuMat img, cv::cuda::GpuMat initial,
                                cv::cuda::GpuMat final, cv::cuda::GpuMat mask,
                                std::string &name, cv::cuda::Stream stream)
{
    displayName = name;
    auto buffer = hostData.getFront();
    img.download(buffer->data[0], stream);
    if(!mask.empty())
        mask.download(buffer->data[1], stream);
    initial.download(buffer->data[2], stream);
    final.download(buffer->data[3], stream);
    buffer->fillEvent.record(stream);
    stream.enqueueHostCallback(FlowVectorDisplay_callback, this);
}*/

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
		std::string displayName = fullTreeName;
		if (d_mask)
		{
			cv::Mat h_initial, h_final, h_mask;
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
	updateParameter("Min value", minVal, Parameters::Parameter::State);
	updateParameter("Min bin", minIdx, Parameters::Parameter::State);
	updateParameter("Max value", maxVal, Parameters::Parameter::State);
	updateParameter("Max bin", maxIdx, Parameters::Parameter::State);
    for(int i = 0; i < data.cols; ++i)
    {
        double height = data.at<int>(i);
        height -= minVal;
        height /=(maxVal - minVal);
        height *= 100;
        cv::rectangle(img, cv::Rect(i*5, 100 - (int)height, 5, 100), cv::Scalar(255),-1);
    }
    cv::imshow(fullTreeName, img);

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
	Parameters::UI::UiCallbackService::Instance()->post(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), fullTreeName, h_img));
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
