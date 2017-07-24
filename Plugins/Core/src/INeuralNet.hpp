#include "Aquila/nodes/IClassifier.hpp"
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include "CoreExport.hpp"
namespace aq{
namespace nodes{
class Core_EXPORT INeuralNet: virtual public  IClassifier{
public:
    MO_DERIVE(INeuralNet, IClassifier)
        OPTIONAL_INPUT(std::vector<cv::Rect2f>, bounding_boxes, nullptr)
        OPTIONAL_INPUT(std::vector<aq::DetectedObject>, input_detections, nullptr)
        INPUT(SyncedMemory, input, nullptr)

        PARAM(mo::ReadFile, model_file, mo::ReadFile())
        TOOLTIP(model_file, "File containing description of neural net")
        PARAM(mo::ReadFile, weight_file, mo::ReadFile())
        TOOLTIP(weight_file, "File containing weights for neural net")
        PARAM(cv::Scalar, channel_mean, cv::Scalar(104, 117, 123))
        TOOLTIP(channel_mean, "Mean BGR pixel values to subtract from input before passing into net")
        PARAM(float, pixel_scale, 0.00390625f)
        TOOLTIP(pixel_scale, "Scale factor to multiply the image by, after mean subtraction")
        PARAM(float, image_scale, 1.0)
        TOOLTIP(image_scale, "Scale factor for input of network. 1.0 = network is resized to input image size, -1.0 = image is resized to network input size")
        PARAM(bool, swap_bgr, true)
    MO_END;
protected:
    virtual bool processImpl();
    virtual std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int requested_batch_size = 1) = 0;
    virtual void preBatch(int batch_size);
    virtual bool forwardAll();
    virtual bool forwardMinibatch() = 0;
    virtual void postBatch(const std::vector<cv::Rect>& batch_bb, const std::vector<DetectedObject2d>& dets) = 0;
    virtual bool reshapeNetwork(int num, int channels, int height, int width) = 0;
};
}
}
