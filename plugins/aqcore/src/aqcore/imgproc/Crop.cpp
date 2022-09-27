#include "Crop.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>
using namespace aq::nodes;

bool Crop::processImpl()
{
    const aq::Shape<3> shape = input->shape();
    cv::Rect2f bb = roi;
    const cv::Size size(shape(0), shape(1));
    boundingBoxToPixels(bb, size);
    aq::SyncedMemory::SyncState state = this->input->state();
    mo::IAsyncStreamPtr_t stream = this->getStream();
    aq::SyncedImage out;
    const aq::PixelFormat pixel_format = this->input->pixelFormat();
    auto input_data = this->input_param.getCurrentData();

    if (state != aq::SyncedMemory::SyncState::DEVICE_UPDATED)
    {
        bool sync = false;
        const cv::Mat mat = input->mat(stream.get(), &sync);
        const cv::Mat roi = mat(bb);
        out = aq::SyncedImage(roi, pixel_format, stream);
        out.setOwning(std::move(input_data));
    }
    else
    {
        if (!stream->isDeviceStream())
        {
            this->getLogger().debug(
                "Provided input stream is not a device stream, whereas the provided input data is on a device");
            return false;
        }
        mo::IDeviceStream::Ptr_t dev_stream = std::dynamic_pointer_cast<mo::IDeviceStream>(stream);
        bool sync = false;
        const cv::cuda::GpuMat mat = input->gpuMat(dev_stream.get(), &sync);
        const cv::cuda::GpuMat roi = mat(bb);
        out = aq::SyncedImage(roi, pixel_format, dev_stream);
        out.setOwning(std::move(input_data));
    }
    this->output.publish(out, mo::tags::param = &input_param);
    return true;
}

MO_REGISTER_CLASS(Crop)
