#include "Channels.h"
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
using namespace aq;
using namespace aq::nodes;


bool ConvertToGrey::processImpl()
{
    if(input_image)
    {
        cv::cuda::GpuMat grey;
        cv::cuda::cvtColor(input_image->getGpuMat(stream()), grey, cv::COLOR_BGR2GRAY, 0, stream());
        grey_image_param.updateData(grey, input_image_param.getTimestamp(), _ctx.get());
        return true;
    }
    return false;
}

bool ConvertToHSV::processImpl()
{
    if (input_image)
    {
        ::cv::cuda::GpuMat hsv;
        ::cv::cuda::cvtColor(input_image->getGpuMat(stream()), hsv, cv::COLOR_BGR2HSV, 0, stream());
        hsv_image_param.updateData(hsv, input_image_param.getTimestamp(), this->_ctx.get());
        return true;
    }
    return false;
}
/*void ConvertColorspace::nodeInit(bool firstInit)
{
    mo::EnumParam param;
    param.addEnum(ENUM(cv::COLOR_BGR2BGRA));
    param.addEnum(ENUM(cv::COLOR_RGB2RGBA));
    param.addEnum(ENUM(cv::COLOR_BGRA2BGR));
    param.addEnum(ENUM(cv::COLOR_RGBA2RGB));
    param.addEnum(ENUM(cv::COLOR_BGR2RGBA));
    param.addEnum(ENUM(cv::COLOR_RGB2BGRA));
    param.addEnum(ENUM(cv::COLOR_RGBA2BGR));
    param.addEnum(ENUM(cv::COLOR_BGRA2RGB));
    param.addEnum(ENUM(cv::COLOR_BGR2RGB));
    param.addEnum(ENUM(cv::COLOR_RGB2BGR));
    param.addEnum(ENUM(cv::COLOR_BGRA2RGBA));
    param.addEnum(ENUM(cv::COLOR_RGBA2BGRA));
    param.addEnum(ENUM(cv::COLOR_BGR2GRAY));
    param.addEnum(ENUM(cv::COLOR_GRAY2BGR));
    param.addEnum(ENUM(cv::COLOR_GRAY2RGB));
    param.addEnum(ENUM(cv::COLOR_GRAY2BGRA));
    param.addEnum(ENUM(cv::COLOR_GRAY2RGBA));
    param.addEnum(ENUM(cv::COLOR_BGRA2GRAY));
    param.addEnum(ENUM(cv::COLOR_RGBA2GRAY));
    param.addEnum(ENUM(cv::COLOR_BGR2BGR565));
    param.addEnum(ENUM(cv::COLOR_RGB2BGR565));
    param.addEnum(ENUM(cv::COLOR_BGR5652BGR));
    param.addEnum(ENUM(cv::COLOR_BGR5652RGB));
    param.addEnum(ENUM(cv::COLOR_BGRA2BGR565));
    param.addEnum(ENUM(cv::COLOR_RGBA2BGR565));
    param.addEnum(ENUM(cv::COLOR_BGR5652BGRA));
    param.addEnum(ENUM(cv::COLOR_BGR5652RGBA));
    param.addEnum(ENUM(cv::COLOR_GRAY2BGR565));

    param.addEnum(ENUM(cv::COLOR_BGR5652GRAY));
    param.addEnum(ENUM(cv::COLOR_BGR2BGR555));
    param.addEnum(ENUM(cv::COLOR_RGB2BGR555));
    param.addEnum(ENUM(cv::COLOR_BGR5552BGR));
    param.addEnum(ENUM(cv::COLOR_BGR5552RGB));
    param.addEnum(ENUM(cv::COLOR_BGRA2BGR555));
    param.addEnum(ENUM(cv::COLOR_RGBA2BGR555));
    param.addEnum(ENUM(cv::COLOR_BGR5552BGRA));
    param.addEnum(ENUM(cv::COLOR_BGR5552RGBA));

    param.addEnum(ENUM(cv::COLOR_GRAY2BGR555));
    param.addEnum(ENUM(cv::COLOR_BGR5552GRAY));

    param.addEnum(ENUM(cv::COLOR_BGR2XYZ));
    param.addEnum(ENUM(cv::COLOR_RGB2XYZ));
    param.addEnum(ENUM(cv::COLOR_XYZ2BGR));
    param.addEnum(ENUM(cv::COLOR_XYZ2RGB));

    param.addEnum(ENUM(cv::COLOR_BGR2YCrCb));
    param.addEnum(ENUM(cv::COLOR_RGB2YCrCb));
    param.addEnum(ENUM(cv::COLOR_YCrCb2BGR));
    param.addEnum(ENUM(cv::COLOR_YCrCb2RGB));

    param.addEnum(ENUM(cv::COLOR_BGR2HSV));
    param.addEnum(ENUM(cv::COLOR_RGB2HSV));

    param.addEnum(ENUM(cv::COLOR_BGR2Lab));
    param.addEnum(ENUM(cv::COLOR_RGB2Lab));

    param.addEnum(ENUM(cv::COLOR_BGR2Luv));
    param.addEnum(ENUM(cv::COLOR_RGB2Luv));
    param.addEnum(ENUM(cv::COLOR_BGR2HLS));
    param.addEnum(ENUM(cv::COLOR_RGB2HLS));

    param.addEnum(ENUM(cv::COLOR_HSV2BGR));
    param.addEnum(ENUM(cv::COLOR_HSV2RGB));

    param.addEnum(ENUM(cv::COLOR_Lab2BGR));
    param.addEnum(ENUM(cv::COLOR_Lab2RGB));
    param.addEnum(ENUM(cv::COLOR_Luv2BGR));
    param.addEnum(ENUM(cv::COLOR_Luv2RGB));
    param.addEnum(ENUM(cv::COLOR_HLS2BGR));
    param.addEnum(ENUM(cv::COLOR_HLS2RGB));

    updateParameter("Conversion Code", param);
}*/
/*cv::cuda::GpuMat ConvertColorspace::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    auto buf =  resultBuffer.getFront();
    cv::cuda::cvtColor(img, buf->data, getParameter<Parameters::EnumParameter>(0)->Data()->getValue(), 0, stream);
    return buf->data;
}*/

bool ConvertTo::processImpl()
{
    if(input->getSyncState() < SyncedMemory::DEVICE_UPDATED )
    {
        cv::Mat output;
        input->getMat(stream()).convertTo(output, datatype.getValue(), alpha, beta);
        this->output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
        return true;
    }else
    {
        cv::cuda::GpuMat output;
        input->getGpuMat(stream()).convertTo(output, datatype.getValue(), alpha, beta, stream());
        this->output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
        return true;
    }
}
MO_REGISTER_CLASS(ConvertTo)

bool Magnitude::processImpl()
{
    if(input_image)
    {
        ::cv::cuda::GpuMat magnitude;
        ::cv::cuda::magnitude(input_image->getGpuMat(stream()), magnitude, stream());
        output_magnitude_param.updateData(magnitude, input_image_param.getTimestamp(), _ctx.get());
        return true;
    }
    return false;
}

bool SplitChannels::processImpl()
{
    if(input_image)
    {
        std::vector<cv::cuda::GpuMat> _channels;
        ::cv::cuda::split(input_image->getGpuMat(stream()), _channels, stream());
        channels_param.updateData(_channels, input_image_param.getTimestamp(), _ctx.get());
        return true;
    }
    return false;
}

bool ConvertDataType::processImpl()
{
    if(input_image)
    {
        ::cv::cuda::GpuMat output;
        if (continuous)
        {
            ::cv::cuda::createContinuous(input_image->getSize(), data_type.currentSelection, output);
        }
        input_image->getGpuMat(stream()).convertTo(output, data_type.currentSelection, alpha, beta, stream());
    }
    return false;
}

/*void Merge::nodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParam<cv::cuda::GpuMat>("Channel1")->SetQualifier(MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
        addInputParam<cv::cuda::GpuMat>("Channel2")->SetQualifier(MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
        addInputParam<cv::cuda::GpuMat>("Channel3")->SetQualifier(MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
        addInputParam<cv::cuda::GpuMat>("Channel4")->SetQualifier(MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
    }
    qualifiersSet = false;
}*/

/*cv::cuda::GpuMat Merge::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    auto chan1 = getParameter<cv::cuda::GpuMat>(0);
    auto chan2 = getParameter<cv::cuda::GpuMat>(1);
    auto chan3 = getParameter<cv::cuda::GpuMat>(2);
    auto chan4 = getParameter<cv::cuda::GpuMat>(3);
//    if(qualifiersSet == false || chan1->changed)
//    {
//        int type = img.type();
//        int width = img.cols;
//        int height = img.rows;
//        boost::function<bool(const Parameter::Ptr&)> f;
//        if(chan1->changed)
//        {
//            if(chan1->data)
//            {
//                type = chan1->data->type();
//                width = chan1->data->cols;
//                height = chan1->data->rows;
//            }
//        }
//        f = GpuMatQualifier::get(width, height, 1, type);
//        updateInputQualifier<cv::cuda::GpuMat>(1,f);
//        updateInputQualifier<cv::cuda::GpuMat>(2,f);
//        updateInputQualifier<cv::cuda::GpuMat>(3,f);
//        _parameters[0]->changed = false;
//        _parameters[1]->changed = false;
//        _parameters[2]->changed = false;
//        _parameters[3]->changed = false;
//        qualifiersSet = true;
//    }
    std::vector<cv::cuda::GpuMat> channels;
    if(chan1->Data())
        channels.push_back(*chan1->Data());
    else
        channels.push_back(img);
    if(chan2->Data())
        channels.push_back(*chan2->Data());
    if(chan3->Data())
        channels.push_back(*chan3->Data());
    if(chan4->Data())
        channels.push_back(*chan4->Data());
    cv::cuda::merge(channels, mergedChannels,stream);
    return mergedChannels;
}*/
bool ConvertColorspace::processImpl()
{
    cv::cuda::GpuMat output;
    cv::cuda::cvtColor(input_image->getGpuMat(stream()),output, conversion_code.getValue(), 0, stream());
    output_image_param.updateData(output, input_image_param.getTimestamp(), _ctx.get());
    return true;
}

bool MergeChannels::processImpl()
{
    return false;
}


bool Reshape::processImpl()
{
    reshaped_image_param.updateData(input_image->getGpuMat(stream()).reshape(channels, rows), input_image_param.getTimestamp(), _ctx.get());
    return true;
}



MO_REGISTER_CLASS(ConvertToGrey)
MO_REGISTER_CLASS(ConvertToHSV)
MO_REGISTER_CLASS(ConvertColorspace)
MO_REGISTER_CLASS(SplitChannels)
MO_REGISTER_CLASS(ConvertDataType)
MO_REGISTER_CLASS(MergeChannels)
MO_REGISTER_CLASS(Reshape)
MO_REGISTER_CLASS(Magnitude)


