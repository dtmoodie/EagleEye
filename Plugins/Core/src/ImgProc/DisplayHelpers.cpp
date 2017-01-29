#include "DisplayHelpers.h"
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
#include <fstream>

using namespace ::EagleLib;
using namespace ::EagleLib::Nodes;

bool Scale::ProcessImpl()
{
    cv::cuda::GpuMat scaled;
    cv::cuda::multiply(input->GetGpuMat(Stream()), cv::Scalar(scale_factor), scaled, 1, -1, Stream());
    output_param.UpdateData(scaled, input_param.GetTimestamp(), _ctx);
    return true;
}
MO_REGISTER_CLASS(Scale)

bool AutoScale::ProcessImpl()
{
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(input_image->GetGpuMat(Stream()), channels, Stream());
    for(size_t i = 0; i < channels.size(); ++i)
    {
        double minVal, maxVal;
        cv::cuda::minMax(channels[i], &minVal, &maxVal);
        double scaleFactor = 255.0 / (maxVal - minVal);
        channels[i].convertTo(channels[0], CV_8U, scaleFactor, minVal*scaleFactor);
        UpdateParameter<double>("Min-" + boost::lexical_cast<std::string>(i), minVal)->SetFlags(mo::State_e);
        UpdateParameter<double>("Max-" + boost::lexical_cast<std::string>(i), maxVal)->SetFlags(mo::State_e);
    }
    cv::cuda::merge(channels,output_image.GetGpuMat(Stream()), Stream());
    return true;
}
bool DrawDetections::ProcessImpl()
{
    if(detection_list_param.modified)
    {
        std::ifstream ifs(detection_list.c_str());
        if(ifs.is_open())
        {
            labels.clear();
            std::string label;
            while(ifs >> label)
                labels.push_back(label);
            colors.resize(labels.size());
            for(int i = 0; i < colors.size(); ++i)
            {
                colors[i] = cv::Vec3b(i * 180 / colors.size(), 200, 255);
            }
            cv::Mat colors_mat(colors.size(), 1, CV_8UC3, &colors[0]);
            cv::cvtColor(colors_mat, colors_mat, cv::COLOR_HSV2BGR);
        }
        detection_list_param.modified = false;
    }
    
    cv::Mat mat_;
    if (image->GetSyncState(0) < SyncedMemory::DEVICE_UPDATED)
    {
        mat_  = image->GetMat(Stream());
    }else
    {
        mat_  = image->GetMat(Stream());
        Stream().waitForCompletion();
    }
    cv::Mat mat = mat_.clone();
    if(detections)
    {
        for(auto& detection : *detections)
        {
            cv::Rect rect(detection.boundingBox.x, detection.boundingBox.y, detection.boundingBox.width, detection.boundingBox.height);
            if(labels.size())
            {
                if(detection.detections.size())
                {
                    cv::rectangle(mat, rect, colors[detection.detections[0].classNumber], 3);
                    std::stringstream ss;
                    if(detection.detections[0].classNumber > 0 && detection.detections[0].classNumber < labels.size())
                    {
                        ss << labels[detection.detections[0].classNumber] << " : " << detection.detections[0].confidence;
                    }else
                    {
                        ss << detection.detections[0].confidence;
                    }
                    cv::putText(mat, ss.str(), rect.tl() + cv::Point(10,20), cv::FONT_HERSHEY_COMPLEX, 0.4, colors[detection.detections[0].classNumber]);
                }
            }else
            {
                // random color for each different detection
                if(detection.detections[0].classNumber >= colors.size())
                {
                    colors.resize(detection.detections[0].classNumber + 1);
                    for(int i = 0; i < colors.size(); ++i)
                    {
                        colors[i] = cv::Vec3b(i * 180 / colors.size(), 200, 255);
                    }
                    cv::Mat colors_mat(colors.size(), 1, CV_8UC3, &colors[0]);
                    cv::cvtColor(colors_mat, colors_mat, cv::COLOR_HSV2BGR);
                }

                cv::rectangle(mat, rect, colors[detection.detections[0].classNumber], 3);
                std::stringstream ss;
                ss << detection.detections[0].classNumber << " : " << detection.detections[0].confidence;
                cv::putText(mat, ss.str(), rect.tl() + cv::Point(10,20), cv::FONT_HERSHEY_COMPLEX, 0.4, colors[detection.detections[0].classNumber]);
            }
        }
    }
    image_with_detections_param.UpdateData(mat, image_param.GetTimestamp(), _ctx);
    return true;
}
/*void
Colormap::NodeInit(bool firstInit)
{
    rescale = true;
    updateParameter("Colormapping scheme", int(0));
    updateParameter<boost::function<void(void)>>("Rescale colormap", boost::bind(&Colormap::Rescale, this));
}
void Colormap::Rescale()
{
    rescale = true;
}

cv::cuda::GpuMat
Colormap::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(img.channels() != 1)
    {
        NODE_LOG(warning) << "Non-monochrome image! Has " + boost::lexical_cast<std::string>(img.channels()) + " channels";
        return img;
    }
    if (rescale)
    {
        double min, max;
        cv::cuda::minMax(img, &min, &max);
        mapper.setMapping(ColorScale(50, 255 / 25, true), ColorScale(50 / 3, 255 / 25, true), ColorScale(0, 255 / 25, true), min, max);
        rescale = false;
    }
    mapper.colormap_image(img, color_mapped_image, stream); 
    return color_mapped_image;
}


QtColormapDisplay::QtColormapDisplay():
    Colormap()
{
}

void QtColormapDisplayCallback(int status, void* data)
{
    QtColormapDisplay* node = static_cast<QtColormapDisplay*>(data);
    Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&QtColormapDisplay::display, node),
        std::make_pair(data, mo::TypeInfo(typeid(EagleLib::Nodes::Node))));
}

void QtColormapDisplay::display()
{

}

void QtColormapDisplay::NodeInit(bool firstInit)
{
    Colormap::Init(firstInit);
}
cv::cuda::GpuMat QtColormapDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(img.channels() != 1)
    {
        NODE_LOG(warning) << "Non-monochrome image! Has " + boost::lexical_cast<std::string>(img.channels()) + " channels";
        return img;
    }
    Colormap::doProcess(img, stream);
    //color_mapped_image.download()

    return img;
}
*/


bool Normalize::ProcessImpl()
{
    cv::cuda::GpuMat normalized;
    
    if(input_image->GetChannels() == 1)
    {
        cv::cuda::normalize(input_image->GetGpuMat(Stream()), 
            normalized,
            alpha,
            beta,
            norm_type.currentSelection, input_image->GetDepth(),
            mask == NULL ? cv::noArray(): mask->GetGpuMat(Stream()),
            Stream());
        normalized_output_param.UpdateData(normalized, input_image_param.GetTimestamp(), _ctx);
        return true;
    }else
    {
        std::vector<cv::cuda::GpuMat> channels;
        
        if (input_image->GetNumMats() == 1)
        {
            cv::cuda::split(input_image->GetGpuMat(Stream()), channels, Stream());
        }else
        {
            channels = input_image->GetGpuMatVec(Stream());
        }
        std::vector<cv::cuda::GpuMat> normalized_channels;
        normalized_channels.resize(channels.size());
        for(int i = 0; i < channels.size(); ++i)
        {
            cv::cuda::normalize(channels[i], normalized_channels,
                alpha,
                beta,
                norm_type.getValue(), input_image->GetDepth(),
                mask == NULL ? cv::noArray() : mask->GetGpuMat(Stream()),
                Stream());
        }
        if(input_image->GetNumMats() == 1)
        {
            cv::cuda::merge(channels, normalized, Stream());
            normalized_output_param.UpdateData(normalized, input_image_param.GetTimestamp(), _ctx);
        }else
        {
            normalized_output_param.UpdateData(normalized_channels, input_image_param.GetTimestamp(), _ctx);
        }
        return true;
    }   
    return false;
}

MO_REGISTER_CLASS(AutoScale)
//NODE_DEFAULT_CONSTRUCTOR_IMPL(Colormap, Image, Processing)
MO_REGISTER_CLASS(Normalize)
MO_REGISTER_CLASS(DrawDetections)
//static EagleLib::Nodes::NodeInfo g_registerer_QtColormapDisplay("QtColormapDisplay", { "Image", "Sink" });
//REGISTERCLASS(QtColormapDisplay, &g_registerer_QtColormapDisplay)
