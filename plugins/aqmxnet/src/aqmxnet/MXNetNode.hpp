#define ALIAS_BOOST_LOG_SEVERITIES
#include <aqmxnet/aqmxnet_export.hpp>

#include <aqcore/INeuralNet.hpp>

#include <MetaObject/core/detail/Enums.hpp>
#include <MetaObject/params/detail/TInputParamPtrImpl.hpp>

#define DMLC_USE_CXX11 1
#define MXNET_USE_CUDA 1
#define MSHADOW_USE_CBLAS 1
#define MSHADOW_USE_CUDA 1
#define MSHADOW_USE_CUDNN 1

#include <mxnet-cpp/model.h>
#include <mxnet/ndarray.h>

namespace aq
{
    namespace mxnet
    {
        class MXNetOutputParser;
    }

    std::vector<std::vector<cv::Mat>> wrapInput(::mxnet::cpp::NDArray& arr, bool swap_rgb = true);
    std::vector<std::vector<cv::cuda::GpuMat>> wrapInputGpu(::mxnet::cpp::NDArray& arr, bool swap_rgb = true);
    cv::Mat wrapOutput(::mxnet::cpp::NDArray& arr);

    namespace nodes
    {

        class aqmxnet_EXPORT MXNet : public INeuralNet
        {
          public:
            MO_DERIVE(MXNet, INeuralNet)
                PARAM(unsigned int, network_width, 600)
                PARAM(unsigned int, network_height, 300)
            MO_END

          protected:
            /// ========================================================
            // overloaded functions
            virtual bool initNetwork() override;
            virtual bool
            reshapeNetwork(unsigned int num, unsigned int channels, unsigned int height, unsigned int width) override;
            virtual cv::Scalar_<unsigned int> getNetworkShape() const override;

            virtual std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int requested_batch_size = 1) override;

            virtual void preBatch(int batch_size) override;
            virtual void postMiniBatch(const std::vector<cv::Rect>& batch_bb, const DetectedObjectSet& dets) override;
            virtual void postBatch() override;
            virtual bool forwardMinibatch() override;

            virtual void addComponent(const rcc::weak_ptr<IAlgorithm>& component) override;
            void addComponent(const rcc::shared_ptr<mxnet::MXNetOutputParser>& component);

            /// ========================================================
            void loadWeights();

            /// ========================================================
            // members
            std::map<std::string, ::mxnet::cpp::NDArray> _args_map; // these are the learned weights and biases
            std::map<std::string, ::mxnet::cpp::NDArray> _aux_map;  // these are additional data members of the layers

            std::unique_ptr<::mxnet::cpp::Executor> _executor;
            ::mxnet::cpp::Symbol _net;
            boost::optional<::mxnet::cpp::Context> _ctx;
            std::map<std::string, std::vector<std::vector<cv::cuda::GpuMat>>> _wrapped_inputs;

            // These map the output of the neural net into a valid output
            std::vector<rcc::shared_ptr<mxnet::MXNetOutputParser>> _parsers;
        }; // class MXNet

    } // namespace nodes
} // namespace aq
