#include <MetaObject/core/metaobject_config.hpp>

#if MO_OPENCV_HAVE_CUDA
#include "Channels.h"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
using namespace aq;
using namespace aq::nodes;

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

bool ConvertTo::processImpl()
{
    if (input->getSyncState() < SyncedMemory::DEVICE_UPDATED)
    {
        cv::Mat output;
        input->getMat(stream()).convertTo(output, datatype.getValue(), alpha, beta);
        this->output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
        return true;
    }
    else
    {
        cv::cuda::GpuMat output;
        input->getGpuMat(stream()).convertTo(output, datatype.getValue(), alpha, beta, stream());
        this->output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
        return true;
    }
}
MO_REGISTER_CLASS(ConvertTo)

MO_REGISTER_CLASS(ConvertColorspace)
MO_REGISTER_CLASS(MergeChannels)

#endif
