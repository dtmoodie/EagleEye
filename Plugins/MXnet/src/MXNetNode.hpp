#define ALIAS_BOOST_LOG_SEVERITIES
#include "MXnetExport.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <MetaObject/params/Types.hpp>
#include <MetaObject/params/detail/TInputParamPtrImpl.hpp>
#define DMLC_USE_CXX11 1
#define MXNET_USE_CUDA 1
#define MSHADOW_USE_CBLAS 1
#define MSHADOW_USE_CUDA 1
#define MSHADOW_USE_CUDNN 1
#include <mxnet-cpp/model.h>
#include <mxnet/ndarray.h>

namespace aq {
class MXNetHandler;
std::vector<std::vector<cv::Mat> > wrapInput(mxnet::cpp::NDArray& arr, bool swap_rgb = true);
std::vector<std::vector<cv::cuda::GpuMat> > wrapInputGpu(mxnet::cpp::NDArray& arr, bool swap_rgb = true);
cv::Mat wrapOutput(mxnet::cpp::NDArray& arr);
namespace nodes {

    class MXnet_EXPORT MXNet : public Node {
    public:
        MO_DERIVE(MXNet, Node)
        INPUT(aq::SyncedMemory, input, nullptr)
        OPTIONAL_INPUT(std::vector<cv::Rect2f>, bounding_boxes, nullptr)
        OPTIONAL_INPUT(std::vector<aq::DetectedObject>, input_detections, nullptr)
        PARAM(cv::Scalar, channel_mean, cv::Scalar(104, 117, 123))
        PARAM(float, pixel_scale, 1.0)
        PARAM(unsigned int, network_width, 600)
        PARAM(unsigned int, network_height, 300)
        PARAM(mo::ReadFile, model_file, mo::ReadFile())
        PARAM(mo::ReadFile, weight_file, mo::ReadFile())
        PARAM(mo::ReadFile, label_file, mo::ReadFile())
        PARAM(bool, swap_bgr, false)
        OUTPUT(std::vector<std::string>, labels, {})
        APPEND_FLAGS(labels, mo::Unstamped_e)
        MO_END;

    protected:
        bool processImpl();
        void postSerializeInit();
        std::map<std::string, mxnet::cpp::NDArray> _args_map; // these are the learned weights and biases
        std::map<std::string, mxnet::cpp::NDArray> _aux_map; // these are additional data members of the layers
        std::unique_ptr<mxnet::cpp::Executor> _executor;
        mxnet::cpp::Symbol                    _net;
        std::map<std::string, mxnet::cpp::NDArray> _inputs;
        mxnet::cpp::NDArray                         _gpu_buffer;
        std::vector<rcc::shared_ptr<MXNetHandler> > _handlers;
    }; // class MXNet

} // namespace aq::nodes
} // namespace aq
