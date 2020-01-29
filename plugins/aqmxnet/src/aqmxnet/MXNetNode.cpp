#include "MXNetNode.hpp"
#include "MXNetOutputParser.hpp"

#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/thread/boost_thread.hpp>

#include <mxnet-cpp/executor.hpp>
#include <mxnet-cpp/io.hpp>
#include <mxnet-cpp/kvstore.hpp>
#include <mxnet-cpp/ndarray.hpp>
#include <mxnet-cpp/symbol.hpp>
#include <mxnet-cpp/symbol.hpp>

#include <boost/filesystem.hpp>

using namespace aq;
using namespace aq::nodes;
// opencv uses BGR ordering, mxnet uses RGB
std::vector<std::vector<cv::Mat>> aq::wrapInput(::mxnet::cpp::NDArray& arr, bool swap_rgb)
{
    std::vector<std::vector<cv::Mat>> output;
    auto shape = arr.GetShape();
    CHECK_EQ(shape.size(), 4);
    unsigned int num = shape[0];
    unsigned int num_channels = shape[1];
    unsigned int height = shape[2];
    unsigned int width = shape[3];
    float* ptr = const_cast<float*>(arr.GetData());
    for (unsigned int i = 0; i < num; ++i)
    {
        std::vector<cv::Mat> channels;
        for (unsigned int i = 0; i < num_channels; ++i)
        {
            channels.push_back(cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_32F, ptr));
            ptr += width * height;
        }
        if (swap_rgb && num_channels == 3)
        {
            std::swap(channels[0], channels[2]);
        }
        output.push_back(channels);
    }
    return output;
}

std::vector<std::vector<cv::cuda::GpuMat>> aq::wrapInputGpu(::mxnet::cpp::NDArray& arr, bool swap_rgb)
{
    std::vector<std::vector<cv::cuda::GpuMat>> output;
    auto shape = arr.GetShape();
    CHECK_EQ(shape.size(), 4);
    int num = static_cast<int>(shape[0]);
    int num_channels = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);
    float* ptr = const_cast<float*>(arr.GetData());
    for (int i = 0; i < num; ++i)
    {
        std::vector<cv::cuda::GpuMat> channels;
        for (int i = 0; i < num_channels; ++i)
        {
            channels.push_back(cv::cuda::GpuMat(height, width, CV_32F, ptr));
            ptr += width * height;
        }
        if (swap_rgb && num_channels == 3)
        {
            std::swap(channels[0], channels[2]);
        }
        output.push_back(channels);
    }
    return output;
}

cv::Mat aq::wrapOutput(::mxnet::cpp::NDArray& arr)
{
    auto shape = arr.GetShape();
    CHECK_EQ(shape.size(), 2);
    int num = static_cast<int>(shape[0]);
    int num_channels = static_cast<int>(shape[1]);
    float* ptr = const_cast<float*>(arr.GetData());
    return cv::Mat(num, num_channels, CV_32F, ptr);
}

void MXNet::loadWeights()
{
    std::map<std::string, ::mxnet::cpp::NDArray> parameters;
    ::mxnet::cpp::NDArray::Load(weight_file.string(), 0, &parameters);
    // upload weights to the GPU... Why is this not automatic? -_-
    for (const auto& k : parameters)
    {
        if (k.first.substr(0, 4) == "aux:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            _aux_map[name] = k.second.Copy(*_ctx); // copy to gpu
        }
        if (k.first.substr(0, 4) == "arg:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            _args_map[name] = k.second.Copy(*_ctx);
        }
    }
}

bool MXNet::initNetwork()
{
    if (!_ctx)
        _ctx = ::mxnet::cpp::Context::gpu();
    if (!_executor || model_file_param.modified() || weight_file_param.modified())
    {
        if (!boost::filesystem::exists(model_file))
        {
            BOOST_LOG_TRIVIAL(warning) << "Model file '" << model_file.string() << "' does not exist!";
            return false;
        }
        if (!boost::filesystem::exists(weight_file))
        {
            BOOST_LOG_TRIVIAL(warning) << "Weight file '" << weight_file.string() << "' does not exist!";
            return false;
        }

        loadWeights();
        _net = ::mxnet::cpp::Symbol::Load(model_file.string());
        if (reshapeNetwork(1, 3, network_height, network_width))
        {
            auto parsers = mxnet::MXNetOutputParser::createParsers(_net);
            for (const auto& parser : parsers)
                addComponent(parser);

            std::vector<::mxnet::cpp::NDArray> outputs;
            mx_uint out_size;
            NDArrayHandle* out_array;
            CHECK_EQ(MXExecutorOutputs(_executor->GetHandle(), &out_size, &out_array), 0);
            for (mx_uint i = 0; i < out_size; ++i)
            {
                outputs.push_back(::mxnet::cpp::NDArray(out_array[i]));
            }

            for (auto& parser : _parsers)
            {
                parser->setupParser(_net, labels, outputs);
            }
            MO_ASSERT(_parsers.size());
            model_file_param.modified(false);
            weight_file_param.modified(false);
            return true;
        }
    }
    return _executor.get();
}

void MXNet::addComponent(const rcc::weak_ptr<IAlgorithm>& component)
{
    rcc::shared_ptr<mxnet::MXNetOutputParser> parser(component);
    if (parser)
    {
        addComponent(parser);
    }
    else
    {
        MO_LOG(info) << "Unable to convert input component of type " << component->GetTypeName()
                     << " to a mxnet::MXNetOutputParser";
    }
}

void MXNet::addComponent(const rcc::shared_ptr<mxnet::MXNetOutputParser>& component)
{
    if (std::find(_parsers.begin(), _parsers.end(), component) == _parsers.end())
    {
        INeuralNet::addComponent(component);
        _parsers.push_back(component);
    }
}

bool MXNet::reshapeNetwork(unsigned int num, unsigned int channels, unsigned int height, unsigned int width)
{
    std::map<std::string, std::vector<mx_uint>> arg_shapes;
    std::vector<std::vector<mx_uint>> in_shape = {{num, channels, height, width}};
    std::vector<std::vector<mx_uint>> out_shape;
    std::vector<std::vector<mx_uint>> aux_shape;
    for (const auto& arg : _args_map)
    {
        arg_shapes[arg.first] = arg.second.GetShape();
    }
    arg_shapes["data"] = {num, channels, height, width};
    auto out = _net.ListOutputs();
    auto args = _net.ListArguments();
    auto data = ::mxnet::cpp::NDArray(::mxnet::cpp::Shape(num, channels, height, width), *_ctx, false);

    _args_map["data"] = data;
    _wrapped_inputs["data"] = wrapInputGpu(data, swap_bgr);
    try
    {
        _executor.reset(_net.SimpleBind(*_ctx,
                                        _args_map,
                                        std::map<std::string, ::mxnet::cpp::NDArray>(),
                                        std::map<std::string, ::mxnet::cpp::OpReqType>(),
                                        _aux_map));
    }
    catch (...)
    {
        MO_LOG(ERROR) << MXGetLastError();
        return false;
    }
    return true;
}

cv::Scalar_<unsigned int> MXNet::getNetworkShape() const
{
    auto itr = _wrapped_inputs.find("data");
    if (itr != _wrapped_inputs.end())
    {
        return {static_cast<unsigned int>(itr->second.size()),
                static_cast<unsigned int>(itr->second[0].size()),
                static_cast<unsigned int>(itr->second[0][0].rows),
                static_cast<unsigned int>(itr->second[0][0].cols)};
    }
    return {};
}

std::vector<std::vector<cv::cuda::GpuMat>> MXNet::getNetImageInput(int /*requested_batch_size*/)
{
    return _wrapped_inputs["data"];
}

void MXNet::preBatch(int batch_size)
{
    for (const auto& parser : _parsers)
    {
        parser->preBatch(static_cast<unsigned int>(batch_size));
    }
}

void MXNet::postMiniBatch(const std::vector<cv::Rect>& batch_bb, const DetectedObjectSet& dets)
{
    try
    {
        for (const auto& parser : _parsers)
        {
            parser->postMiniBatch(batch_bb, dets);
        }
    }
    catch (const dmlc::Error& err)
    {
        MO_LOG(ERROR) << MXGetLastError();
        MO_LOG(ERROR) << err.what();
    }
}

void MXNet::postBatch()
{
    for (const auto& parser : _parsers)
    {
        parser->postBatch(input_param);
    }
}

bool MXNet::forwardMinibatch()
{
    if (_executor)
    {
        stream().waitForCompletion();
        _executor->Forward(false);
        return true;
    }
    return false;
}

MO_REGISTER_CLASS(MXNet)
