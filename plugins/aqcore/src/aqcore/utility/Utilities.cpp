#include "Utilities.h"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>

#include <boost/lexical_cast.hpp>

namespace aqcore
{

    bool RegionOfInterest::processImpl()
    {
        if (roi.area())
        {
            // auto img_roi = cv::Rect2f(cv::Point2f(0.0,0.0), image->getSize());
            const cv::Rect2f img_roi = cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f);
            cv::Rect2f used_roi = img_roi & roi;

            const aq::Shape<2> img_shape = input->size();
            const cv::Size img_size(img_shape(0), img_shape(1));

            aq::boundingBoxToPixels(used_roi, img_size);
            const cv::Rect pixel_roi = cv::Rect(used_roi) & cv::Rect(cv::Point(), img_size);

            mo::IAsyncStreamPtr_t stream = this->getStream();
            mo::IDeviceStream* dev_stream = stream->getDeviceStream();
            const aq::SyncedMemory::SyncState state = input->state();
            aq::SyncedImage output_image;
            if (state == aq::SyncedMemory::SyncState::HOST_UPDATED)
            {
                cv::Mat mat = input->getMat(stream.get());
                cv::Mat roi = mat(pixel_roi);
                output_image = aq::SyncedImage(roi);
            }
            else
            {
                cv::cuda::GpuMat mat = input->getGpuMat(dev_stream);
                cv::cuda::GpuMat roi = mat(pixel_roi);
                output_image = aq::SyncedImage(roi);
            }

            output.publish(std::move(output_image), mo::tags::param = &input_param);
            return true;
        }
        return false;
    }

    void ExportRegionsOfInterest::nodeInit(bool firstInit)
    {
        output.setName("output");
        output.setFlags(mo::ParamFlags::kOUTPUT);
        output.appendFlags(mo::ParamFlags::kUNSTAMPED);
        output.publish(rois);
        addParam(output);
    }

    bool ExportRegionsOfInterest::processImpl() { return true; }
} // namespace aqcore

using namespace aqcore;
MO_REGISTER_CLASS(ExportRegionsOfInterest)
MO_REGISTER_CLASS(RegionOfInterest)