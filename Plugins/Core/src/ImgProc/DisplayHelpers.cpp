#include "DisplayHelpers.h"
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
using namespace ::EagleLib;
using namespace ::EagleLib::Nodes;


bool AutoScale::ProcessImpl()
{

    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(input_image->GetGpuMat(*_ctx->stream), channels, *_ctx->stream);
    for(size_t i = 0; i < channels.size(); ++i)
    {
        double minVal, maxVal;
        cv::cuda::minMax(channels[i], &minVal, &maxVal);
        double scaleFactor = 255.0 / (maxVal - minVal);
        channels[i].convertTo(channels[0], CV_8U, scaleFactor, minVal*scaleFactor);
        UpdateParameter<double>("Min-" + boost::lexical_cast<std::string>(i), minVal)->SetFlags(mo::State_e);
        UpdateParameter<double>("Max-" + boost::lexical_cast<std::string>(i), maxVal)->SetFlags(mo::State_e);
    }
    cv::cuda::merge(channels,output_image.GetGpuMat(*_ctx->stream), *_ctx->stream);
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
        cv::cuda::normalize(input_image->GetGpuMat(*_ctx->stream), 
            normalized,
            alpha,
            beta,
            norm_type.currentSelection, input_image->GetDepth(),
            mask == NULL ? cv::noArray(): mask->GetGpuMat(*_ctx->stream),
            *_ctx->stream);
        normalized_output_param.UpdateData(normalized, input_image_param.GetTimestamp(), _ctx);
        return true;
    }else
    {
        std::vector<cv::cuda::GpuMat> channels;
        
        if (input_image->GetNumMats() == 1)
        {
            cv::cuda::split(input_image->GetGpuMat(*_ctx->stream), channels, *_ctx->stream);
        }else
        {
            channels = input_image->GetGpuMatVec(*_ctx->stream);
        }
        std::vector<cv::cuda::GpuMat> normalized_channels;
        normalized_channels.resize(channels.size());
        for(int i = 0; i < channels.size(); ++i)
        {
            cv::cuda::normalize(channels[i], normalized_channels,
                alpha,
                beta,
                norm_type.getValue(), input_image->GetDepth(),
                mask == NULL ? cv::noArray() : mask->GetGpuMat(*_ctx->stream),
                *_ctx->stream);
        }
        if(input_image->GetNumMats() == 1)
        {
            cv::cuda::merge(channels, normalized, *_ctx->stream);
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
MO_REGISTER_CLASS(Normalize, Image, Processing)
//static EagleLib::Nodes::NodeInfo g_registerer_QtColormapDisplay("QtColormapDisplay", { "Image", "Sink" });
//REGISTERCLASS(QtColormapDisplay, &g_registerer_QtColormapDisplay)
