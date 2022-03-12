#pragma once
#include "IClassifier.hpp"

#include "aqcore/aqcore_export.hpp"

#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedImage.hpp>

namespace aqcore
{

    struct INeuralNetInfo : public aq::nodes::NodeInfo
    {
        virtual bool canLoad(const std::string& model, const std::string& weights) const = 0;
    };

    class aqcore_EXPORT INeuralNet : virtual public TInterface<INeuralNet, IClassifier>

    {
      public:
        static rcc::shared_ptr<INeuralNet> create(const std::string& model, const std::string& weight = std::string());

        MO_DERIVE(INeuralNet, IClassifier)
            INPUT(aq::SyncedImage, input)
            OPTIONAL_INPUT(mo::vector<cv::Rect2f>, regions_of_interest)
            OPTIONAL_INPUT(aq::TDetectedObjectSet<ct::VariadicTypedef<aq::detection::BoundingBox2d>>, input_detections)

            PARAM(mo::ReadFile, model_file, mo::ReadFile())

            PARAM(mo::ReadFile, weight_file, mo::ReadFile())
            PARAM_UPDATE_SLOT(weight_file)

            PARAM(cv::Scalar, channel_mean, cv::Scalar(104, 117, 123))

            PARAM(mo::ReadFile, mean_file, {})

            PARAM(float, pixel_scale, 0.00390625f)

            PARAM(float, image_scale, 1.0)

            PARAM(bool, swap_bgr, true)
        MO_END;

      protected:
        bool processImpl(mo::IAsyncStream& stream) override;
        bool processImpl(mo::IDeviceStream& stream) override;

        void setStream(const mo::IAsyncStreamPtr_t& stream) override;

        virtual bool initNetwork() = 0;

        virtual bool
        reshapeNetwork(unsigned int num, unsigned int channels, unsigned int height, unsigned int width) = 0;

        virtual cv::Scalar_<unsigned int> getNetworkShape() const = 0;

        virtual std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int requested_batch_size = 1) = 0;

        virtual void preBatch(int batch_size);

        virtual void postMiniBatch(mo::IDeviceStream& stream,
                                   const std::vector<cv::Rect>& batch_bb = std::vector<cv::Rect>(),
                                   const aq::DetectedObjectSet* dets = nullptr) = 0;
        virtual void postBatch();

        virtual bool forwardAll(mo::IDeviceStream& stream);

        virtual bool forwardMinibatch(mo::IDeviceStream& stream) = 0;
        // Return a list of pixel coordinates in the input image for processing
        std::vector<cv::Rect> getRegions() const;
        std::unique_ptr<cv::cuda::Stream> m_cv_stream;
    };

} // namespace aqcore
