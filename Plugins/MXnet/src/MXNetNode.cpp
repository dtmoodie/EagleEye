#include "MXNetNode.hpp"
#include "MXNetHandler.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/thread/boost_thread.hpp>
#include <mxnet-cpp/executor.hpp>
#include <mxnet-cpp/io.hpp>
#include <mxnet-cpp/kvstore.hpp>
#include <mxnet-cpp/ndarray.hpp>
#include <mxnet-cpp/symbol.hpp>
#include <mxnet-cpp/symbol.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace aq;
using namespace aq::nodes;
// opencv uses BGR ordering, mxnet uses RGB
std::vector<std::vector<cv::Mat>> aq::wrapInput(mxnet::cpp::NDArray& arr, bool swap_rgb)
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

std::vector<std::vector<cv::cuda::GpuMat>> aq::wrapInputGpu(mxnet::cpp::NDArray& arr, bool swap_rgb)
{
    std::vector<std::vector<cv::cuda::GpuMat>> output;
    auto shape = arr.GetShape();
    CHECK_EQ(shape.size(), 4);
    int num = static_cast<int>(shape[0]);
    int num_channels = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);
    float* ptr = const_cast<float*>(arr.GetDataGpu());
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

cv::Mat aq::wrapOutput(mxnet::cpp::NDArray& arr)
{
    auto shape = arr.GetShape();
    CHECK_EQ(shape.size(), 2);
    int num = static_cast<int>(shape[0]);
    int num_channels = static_cast<int>(shape[1]);
    float* ptr = const_cast<float*>(arr.GetData());
    return cv::Mat(num, num_channels, CV_32F, ptr);
}

// TODO batch size based on input rois
// TODO GPU buffer instead of cpu buffer
bool MXNet::processImpl()
{
    auto gpu_ctx = mxnet::cpp::Context::gpu();

    if ((label_file_param.modified() || labels.empty()) && boost::filesystem::exists(label_file))
    {
        labels.clear();
        std::ifstream ifs(label_file.string().c_str());

        std::string line;
        while (std::getline(ifs, line, '\n'))
        {
            labels.push_back(line);
        }
        BOOST_LOG_TRIVIAL(info) << "Loaded " << labels.size() << " classes";
        labels_param.emitUpdate();
        label_file_param.modified(false);
    }
    unsigned int batch_size = 5;
    if (!_executor || bounding_boxes_param.modified())
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
        if (bounding_boxes)
        {
            batch_size = static_cast<unsigned int>(bounding_boxes->size());
        }
        uint num_ops;
        const char** out_names;
        MXListAllOpNames(&num_ops, &out_names); // https://github.com/dmlc/mxnet/pull/4537

        std::map<std::string, mxnet::cpp::NDArray> parameters;
        mxnet::cpp::NDArray::Load(weight_file.string(), 0, &parameters);
        // upload weights to the GPU... Why is this not automatic? -_-
        for (const auto& k : parameters)
        {
            if (k.first.substr(0, 4) == "aux:")
            {
                auto name = k.first.substr(4, k.first.size() - 4);
                _aux_map[name] = k.second.Copy(gpu_ctx); // copy to gpu
            }
            if (k.first.substr(0, 4) == "arg:")
            {
                auto name = k.first.substr(4, k.first.size() - 4);
                _args_map[name] = k.second.Copy(gpu_ctx);
            }
        }
        mxnet::cpp::Symbol sym = mxnet::cpp::Symbol::Load(model_file.string());
        std::map<std::string, std::vector<mx_uint>> arg_shapes;
        std::vector<std::vector<mx_uint>> in_shape = {{batch_size, 3, network_height, network_width}};
        std::vector<std::vector<mx_uint>> out_shape;
        std::vector<std::vector<mx_uint>> aux_shape;
        for (const auto& arg : _args_map)
        {
            arg_shapes[arg.first] = arg.second.GetShape();
        }
        arg_shapes["data"] = {batch_size, 3, network_height, network_width};
        sym.InferShape(arg_shapes, &in_shape, &aux_shape, &out_shape);
        auto out = sym.ListOutputs();
        auto args = sym.ListArguments();
        if (out.size() == 1)
            _net = sym.GetInternals()[out[0]];
        auto data =
            mxnet::cpp::NDArray(mxnet::cpp::Shape(batch_size, 3, network_height, network_width), gpu_ctx, false);
        _args_map["data"] = data;
        _inputs["data"] = data;
        try
        {
            // my assumption is that this is where mxnet instantiates its thread pool
            std::string old_name = mo::getThisThreadName();
            mo::setThisThreadName("mxnet");
            _executor.reset(_net.SimpleBind(gpu_ctx,
                                            _args_map,
                                            std::map<std::string, mxnet::cpp::NDArray>(),
                                            std::map<std::string, mxnet::cpp::OpReqType>(),
                                            _aux_map));
            mo::setThisThreadName(old_name);
        }
        catch (...)
        {
            MO_LOG(ERROR) << MXGetLastError();
            return false;
        }
        _gpu_buffer = data;
        //_gpu_buffer = mxnet::cpp::NDArray(mxnet::cpp::Shape(batch_size,3,network_height, network_width), gpu_ctx,
        //false);
        bounding_boxes_param.modified(false);
    }
    if (!_executor)
    {
        if (!boost::filesystem::exists(model_file))
        {
            MO_LOG(WARNING) << "NN model file '" << model_file.string() << "' does not exists!";
        }
        if (!boost::filesystem::exists(weight_file))
        {
            MO_LOG(WARNING) << "NN weight file '" << weight_file.string() << "' does not exists!";
        }
        return false;
    }
    auto input_image_shape = input->getShape();
    std::vector<cv::Rect2f> default_roi(1, cv::Rect2f(0, 0, 1.0, 1.0));
    if (bounding_boxes == nullptr)
        bounding_boxes = &default_roi;
    if (input_detections != nullptr && bounding_boxes == &default_roi)
    {
        default_roi.clear();
        for (const auto& itr : *input_detections)
        {
            default_roi.emplace_back(itr.boundingBox.x / float(input_image_shape[2]),
                                     itr.boundingBox.y / float(input_image_shape[1]),
                                     itr.boundingBox.width / float(input_image_shape[2]),
                                     itr.boundingBox.height / float(input_image_shape[1]));
        }
        if (default_roi.size() == 0)
        {
            for (auto& handler : _handlers)
            {
                handler->startBatch();
                handler->endBatch(input_param);
            }
            if (bounding_boxes == &default_roi)
            {
                bounding_boxes = nullptr;
            }
            return false;
        }
    }

    std::vector<cv::Rect> pixel_bounding_boxes;
    for (size_t i = 0; i < bounding_boxes->size(); ++i)
    {
        cv::Rect bb;
        bb.x = static_cast<int>((*bounding_boxes)[i].x * input_image_shape[2]);
        bb.y = static_cast<int>((*bounding_boxes)[i].y * input_image_shape[1]);
        bb.width = static_cast<int>((*bounding_boxes)[i].width * input_image_shape[2]);
        bb.height = static_cast<int>((*bounding_boxes)[i].height * input_image_shape[1]);
        if (bb.x + bb.width >= input_image_shape[2])
        {
            bb.x -= input_image_shape[2] - bb.width;
        }
        if (bb.y + bb.height >= input_image_shape[1])
        {
            bb.y -= input_image_shape[1] - bb.height;
        }
        bb.x = std::max(0, bb.x);
        bb.y = std::max(0, bb.y);
        pixel_bounding_boxes.push_back(bb);
    }
    cv::cuda::GpuMat d_in = input->getGpuMat(stream());
    cv::cuda::GpuMat float_img;
    if (d_in.depth() != CV_32F)
    {
        d_in.convertTo(float_img, CV_32F, stream());
    }
    else
    {
        float_img = d_in;
    }
    cv::cuda::subtract(float_img, channel_mean, float_img, cv::noArray(), -1, stream());
    if (pixel_scale != 1.0f)
    {
        cv::cuda::multiply(float_img, static_cast<double>(pixel_scale), float_img, 1.0, -1, stream());
    }
    // for testing resize whole input image and shove into network

    mxnet::cpp::NDArray input = _inputs["data"];
    auto wrapped = wrapInputGpu(_gpu_buffer, swap_bgr);
    cv::Size nn_input_size = wrapped[0][0].size();
    for (auto& handler : _handlers)
        handler->startBatch();
    for (size_t i = 0; i < pixel_bounding_boxes.size();)
    {
        cv::cuda::GpuMat resized;
        size_t start = i, end = 0;
        for (size_t j = 0; j < wrapped.size() && i < pixel_bounding_boxes.size(); ++j, ++i)
        {
            if (pixel_bounding_boxes[i].size() != nn_input_size)
            {
                cv::cuda::resize(
                    float_img(pixel_bounding_boxes[i]), resized, nn_input_size, 0, 0, cv::INTER_LINEAR, stream());
            }
            else
            {
                resized = float_img(pixel_bounding_boxes[i]);
            }
            cv::cuda::split(resized, wrapped[j], stream());
            end = start + j + 1;
        }
        stream().waitForCompletion(); // TODO asyc copy into mxnet... Need to expand their api
        _gpu_buffer.CopyTo(&input);
        {
            mo::scoped_profile profile_forward("MXNET Neural Net forward pass", nullptr, nullptr, cudaStream());
            _executor->Forward(false);
            MXNetHandler::OutputMapping_t outmap;
            auto output_symbols = _net.ListOutputs();
            for (size_t i = 0; i < output_symbols.size(); ++i)
            {
                outmap[output_symbols[i]].push_back(_executor->outputs[i]);
            }
            if (_handlers.size() == 0)
            {
                auto handlers = MXNetHandler::Create(_net);
                if (handlers.size() == 0)
                {
                    auto dbgprint = [&outmap]() {
                        std::stringstream ss;
                        for (const auto& itr : outmap)
                        {
                            ss << itr.first << ", ";
                        }
                        return ss.str();
                    };
                    MO_LOG(WARNING) << "Unable to create network handler for network with outputs: " << dbgprint();
                }
                for (auto& handler : handlers)
                {
                    handler->setContext(this->getContext());
                    handler->setLabels(&this->labels);
                    this->_algorithm_components.emplace_back(handler);
                    handler->startBatch();
                }
                _handlers.insert(_handlers.end(), handlers.begin(), handlers.end());
            }
            std::vector<aq::DetectedObject2d> dets;
            std::vector<cv::Rect> bbs;
            if (input_detections != nullptr && bounding_boxes == &default_roi)
            {
                for (size_t j = start; j < end; ++j)
                {
                    dets.push_back((*input_detections)[j]);
                }
            }
            for (size_t j = start; j < end; ++j)
            {
                bbs.push_back(pixel_bounding_boxes[j]);
            }
            for (auto& handler : _handlers)
            {
                handler->handleMiniBatch(outmap, bbs, dets, input_param);
            }
        }
    }
    for (auto& handler : _handlers)
    {
        handler->endBatch(input_param);
    }
    // mxnet::cpp::NDArray out = executor->outputs[0].Copy(cpu_ctx);
    if (bounding_boxes == &default_roi)
    {
        bounding_boxes = nullptr;
    }
    return true;
}

void MXNet::postSerializeInit()
{
    Node::postSerializeInit();
    for (auto& component : _algorithm_components)
    {
        rcc::shared_ptr<MXNetHandler> handler(component);
        if (handler)
        {
            _handlers.push_back(handler);
            handler->setLabels(&this->labels);
            handler->setContext(this->getContext());
        }
    }
}

MO_REGISTER_CLASS(MXNet)
