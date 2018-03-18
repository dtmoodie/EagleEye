#include "IClassifier.hpp"
#include "aqcore_export.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aq
{
    namespace nodes
    {
        struct INeuralNetInfo : public NodeInfo
        {
            virtual bool canLoad(const std::string& model, const std::string& weights) = 0;
        };

        class aqcore_EXPORT INeuralNet : virtual public TInterface<INeuralNet, IClassifier>
        {
          public:
            static rcc::shared_ptr<INeuralNet> create(const std::string& model,
                                                      const std::string& weight = std::string());

            MO_DERIVE(INeuralNet, IClassifier)
                INPUT(SyncedMemory, input, nullptr)
                OPTIONAL_INPUT(std::vector<cv::Rect2f>, bounding_boxes, nullptr)
                OPTIONAL_INPUT(DetectedObjectSet, input_detections, nullptr)

                PARAM(mo::ReadFile, model_file, mo::ReadFile())
                TOOLTIP(model_file, "File containing description of neural net")

                PARAM(mo::ReadFile, weight_file, mo::ReadFile())
                TOOLTIP(weight_file, "File containing weights for neural net")
                PARAM_UPDATE_SLOT(weight_file)

                PARAM(cv::Scalar, channel_mean, cv::Scalar(104, 117, 123))
                TOOLTIP(channel_mean, "Mean BGR pixel values to subtract from input before passing into net")

                PARAM(mo::ReadFile, mean_file, {})
                TOOLTIP(mean_file, "File containing a per pixel mean value to subtract from the input")

                PARAM(float, pixel_scale, 0.00390625f)
                TOOLTIP(pixel_scale, "Pixel value scale to multiply image by after subtraction")

                PARAM(float, image_scale, 1.0)
                TOOLTIP(image_scale,
                        "Scale factor for input of network. 1.0 = network is resized to input image size, "
                        "-1.0 = image is resized to network input size")

                PARAM(bool, swap_bgr, true)
            MO_END

          protected:
            virtual bool processImpl() override;

            virtual bool initNetwork() = 0;
            virtual bool
            reshapeNetwork(unsigned int num, unsigned int channels, unsigned int height, unsigned int width) = 0;
            virtual cv::Scalar_<unsigned int> getNetworkShape() const = 0;

            virtual std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int requested_batch_size = 1) = 0;

            virtual void preBatch(int batch_size);
            virtual void postMiniBatch(const std::vector<cv::Rect>& batch_bb = std::vector<cv::Rect>(),
                                       const DetectedObjectSet& dets = DetectedObjectSet()) = 0;
            virtual void postBatch();

            virtual bool forwardAll();
            virtual bool forwardMinibatch() = 0;
        };
    }
}
