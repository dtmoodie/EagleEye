#include "MaskDisplay.hpp"
#include "Aquila/nodes/NodeInfo.hpp"

namespace aqcore
{
    bool MaskOverlay::processImpl()
    {
        const aq::Shape<3> img_shape = image->shape();
        const aq::Shape<3> mask_shape = mask->shape();
        MO_ASSERT((img_shape == mask_shape).all());

        const aq::SyncedMemory::SyncState img_state = image->state();
        const aq::SyncedMemory::SyncState mask_state = mask->state();
        const bool img_on_device = img_state != aq::SyncedMemory::SyncState::HOST_UPDATED;
        const bool mask_on_device = mask_state != aq::SyncedMemory::SyncState::HOST_UPDATED;

        if (!img_on_device && !mask_on_device)
        {
            // Since both are on the host, just do this on the host

            return true;
        }

        // device side
        mo::IAsyncStreamPtr_t stream = this->getStream();
        mo::IDeviceStream* dev_stream = stream->getDeviceStream();
        const aq::PixelFormat pixel_format = image->pixelFormat();
        const aq::DataFlag data_flag = image->dataType();
        cv::cuda::Stream* cv_stream = this->getCVStream();
        MO_ASSERT(cv_stream != nullptr);

        aq::SyncedImage output(aq::Shape<2>(img_shape(0), img_shape(1)), pixel_format, data_flag, stream);
        cv::cuda::GpuMat gpu_out = output.gpuMat();
        image->copyTo(gpu_out, stream->getDeviceStream());

        cv::cuda::GpuMat gpu_mask = mask->gpuMat(dev_stream);

        gpu_out.setTo(color, gpu_mask, *cv_stream);
        this->output.publish(std::move(output), mo::tags::param = &image_param);

        return true;
    }

    const cv::Scalar MaskOverlay::default_color = cv::Scalar(255, 0, 0, 0);

} // namespace aqcore

using namespace aqcore;

MO_REGISTER_CLASS(MaskOverlay)
