#define PARAMTERS_GENERATE_PERSISTENCE
#include "Caffe.h"
#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include "caffe_include.h"
#include "caffe_init.h"
#include "helpers.hpp"
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <MetaObject/params/Types.hpp>
#include <boost/tokenizer.hpp>

#include <string>

#include "caffe/caffe.hpp"

using namespace aq;
using namespace aq::nodes;
#ifndef _MSC_VER
#include "dlfcn.h"
#else

#endif
void initModule(SystemTable*)
{
#ifndef _MSC_VER
    dlopen("libpython2.7.so", RTLD_LAZY | RTLD_GLOBAL);
#endif
}

std::vector<SyncedMemory> CaffeImageClassifier::WrapBlob(caffe::Blob<float>& blob, bool bgr_swap)
{
    std::vector<SyncedMemory> wrapped_blob;
    int height = blob.height();
    int width = blob.width();
    float* h_ptr = blob.mutable_cpu_data();
    float* d_ptr = blob.mutable_gpu_data();
    for (int j = 0; j < blob.num(); ++j)
    {
        std::vector<cv::cuda::GpuMat> d_wrappedChannels;
        std::vector<cv::Mat> h_wrappedChannels;
        for (int i = 0; i < blob.channels(); ++i)
        {
            cv::cuda::GpuMat d_channel(height, width, CV_32FC1, d_ptr);
            cv::Mat h_channel(height, width, CV_32F, h_ptr);
            d_wrappedChannels.push_back(d_channel);
            h_wrappedChannels.push_back(h_channel);
            d_ptr += height * width;
            h_ptr += height * width;
        }
        if (bgr_swap && h_wrappedChannels.size() == 3 && d_wrappedChannels.size() == 3)
        {
            std::swap(h_wrappedChannels[0], h_wrappedChannels[2]);
            std::swap(d_wrappedChannels[0], d_wrappedChannels[2]);
        }
        SyncedMemory image(h_wrappedChannels, d_wrappedChannels, SyncedMemory::DO_NOT_SYNC);
        wrapped_blob.push_back(image);
    }
    return wrapped_blob;
}

std::vector<SyncedMemory> CaffeImageClassifier::WrapBlob(caffe::Blob<double>& blob, bool bgr_swap)
{
    std::vector<SyncedMemory> wrapped_blob;
    int height = blob.height();
    int width = blob.width();
    double* d_ptr = blob.mutable_gpu_data();
    double* h_ptr = blob.mutable_cpu_data();
    for (int j = 0; j < blob.num(); ++j)
    {
        std::vector<cv::cuda::GpuMat> d_wrappedChannels;
        std::vector<cv::Mat> h_wrappedChannels;
        for (int i = 0; i < blob.channels(); ++i)
        {
            cv::cuda::GpuMat d_channel(height, width, CV_64FC1, d_ptr);
            cv::Mat h_channel(height, width, CV_64F, h_ptr);
            d_wrappedChannels.push_back(d_channel);
            h_wrappedChannels.push_back(h_channel);
            d_ptr += height * width;
            h_ptr += height * width;
        }
        if (bgr_swap && h_wrappedChannels.size() == 3 && d_wrappedChannels.size() == 3)
        {
            std::swap(h_wrappedChannels[0], h_wrappedChannels[2]);
            std::swap(d_wrappedChannels[0], d_wrappedChannels[2]);
        }
        SyncedMemory image(h_wrappedChannels, d_wrappedChannels, SyncedMemory::DO_NOT_SYNC);
        wrapped_blob.push_back(image);
    }
    return wrapped_blob;
}

void CaffeImageClassifier::WrapInput()
{
    if (NN == nullptr)
    {
        MO_LOG_EVERY_N(error, 100) << "Neural network not defined";
        return;
    }
    if (NN->num_inputs() == 0)
        return;
    auto input_blob_indecies = NN->input_blob_indices();
    std::vector<std::string> input_names;
    for (auto idx : input_blob_indecies)
    {
        input_names.push_back(NN->blob_names()[idx]);
    }
    input_blobs = NN->input_blobs();

    std::stringstream ss;
    ss << "Architecture loaded, num inputs: " << NN->num_inputs();
    ss << " num outputs: " << NN->num_outputs() << "\n";
    for (int i = 0; i < input_blobs.size(); ++i)
    {
        ss << "   input batch size: " << input_blobs[i]->num() << "\n";
        ss << "   input channels: " << input_blobs[i]->channels() << "\n";
        ss << "   input size: (" << input_blobs[i]->width() << ", " << input_blobs[i]->height() << ")\n";
    }
    // MO_LOG(debug) << ss.str();

    for (int k = 0; k < input_blobs.size(); ++k)
    {
        wrapped_inputs[input_names[k]] = WrapBlob(*input_blobs[k], swap_bgr);
    }
}

bool CaffeImageClassifier::CheckInput()
{
    if (NN == nullptr)
        return false;
    const auto& input_blob_indecies = NN->input_blob_indices();
    std::vector<std::string> input_names;
    for (auto idx : input_blob_indecies)
    {
        input_names.push_back(NN->blob_names()[idx]);
    }
    auto input_blobs_ = NN->input_blobs();
    if (input_blobs_.size() != input_blobs.size())
        return false;
    for (int i = 0; i < input_blob_indecies.size(); ++i)
    {
        auto itr = wrapped_inputs.find(input_names[i]);
        if (itr != wrapped_inputs.end())
        {
            const float* data = input_blobs[i]->gpu_data();
            for (int j = 0; j < itr->second.size(); ++j)
            {
                for (int k = 0; k < itr->second[j].getNumMats(); ++k)
                {
                    const cv::cuda::GpuMat& mat = itr->second[j].getGpuMat(stream(), k);
                    if (data != (float*)mat.data)
                    {
                        return false;
                    }
                    data += mat.rows * mat.cols;
                }
            }
        }
    }
    return true;
}

bool CaffeImageClassifier::reshapeNetwork(unsigned int num,
                                          unsigned int channels,
                                          unsigned int height,
                                          unsigned int width)
{
    input_blobs = NN->input_blobs();
    for (auto input_blob : input_blobs)
    {
        input_blob->Reshape(num, channels, height, width);
    }
    if (!CheckInput())
        WrapInput();
    return true;
}

void CaffeImageClassifier::WrapOutput()
{
    if (NN == nullptr)
    {
        BOOST_LOG_TRIVIAL(error) << "Neural network not defined";
        return;
    }
    if (NN->num_inputs() == 0)
        return;

    auto outputs = NN->output_blobs();
    wrapped_outputs.clear();
    auto output_idx = NN->output_blob_indices();
    wrapped_outputs.clear();
    for (int i = 0; i < output_idx.size(); ++i)
    {
        wrapped_outputs[NN->blob_names()[output_idx[i]]] = WrapBlob(*outputs[i]);
    }
    auto layers = NN->layers();
    bool has_fully_connected = false;
    for (auto layer : layers)
    {
        if (layer->type() == std::string("InnerProduct"))
        {
            has_fully_connected = true;
        }
    }
}

bool CaffeImageClassifier::initNetwork()
{
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
    if (model_file_param.modified())
    {
        if (boost::filesystem::exists(model_file))
        {
            std::string param_file = model_file.string();
            try
            {
                NN.reset(new caffe::Net<float>(param_file, caffe::TEST));
            }
            catch (caffe::ExceptionWithCallStack<std::string>& exp)
            {
                throw mo::ExceptionWithCallStack<std::string>(exp, exp.CallStack());
            }
            WrapInput();
            WrapOutput();
            model_file_param.modified(false);
        }
        else
        {
            MO_LOG_EVERY_N(warning, 100) << "Architecture file does not exist " << model_file.string();
        }
    }

    if (weight_file_param.modified() && NN)
    {
        if (boost::filesystem::exists(weight_file))
        {
            try
            {
                NN->CopyTrainedLayersFrom(weight_file.string());
            }
            catch (caffe::ExceptionWithCallStack<std::string>& exp)
            {
                throw mo::ExceptionWithCallStack<std::string>(exp, exp.CallStack());
            }
            catch (...)
            {
                return false;
            }
            const std::vector<boost::shared_ptr<caffe::Layer<float>>>& layers = NN->layers();
            std::vector<std::string> layerNames;
            layerNames.reserve(layers.size());
            for (auto layer : layers)
            {
                layerNames.push_back(std::string(layer->type()));
            }
            BOOST_LOG_TRIVIAL(info) << "Weights loaded";
            weightsLoaded = true;
            // UpdateParameter("Loaded layers", layerNames);
            weight_file_param.modified(false);
        }
        else
        {
            MO_LOG_EVERY_N(warning, 100) << "Weight file does not exist " << weight_file.string();
        }
    }

    if ((label_file_param.modified() || labels.empty()) && boost::filesystem::exists(label_file))
    {
        labels.clear();
        std::ifstream ifs(label_file.string().c_str());
        if (!ifs)
        {
            MO_LOG_EVERY_N(warning, 100) << "Unable to load label file";
        }

        std::string line;
        while (std::getline(ifs, line, '\n'))
        {
            labels.push_back(line);
        }
        BOOST_LOG_TRIVIAL(info) << "Loaded " << labels.size() << " classes";
        labels_param.emitUpdate();
        label_file_param.modified(false);
    }

    if (mean_file_param.modified())
    {
        if (boost::filesystem::exists(mean_file))
        {
            if (boost::filesystem::is_regular_file(mean_file))
            {
                caffe::BlobProto blob_proto;
                if (caffe::ReadProtoFromBinaryFile(mean_file.string().c_str(), &blob_proto))
                {
                    caffe::Blob<float> mean_blob;
                    mean_blob.FromProto(blob_proto);
                    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
                    std::vector<cv::Mat> channels;
                    float* data = mean_blob.mutable_cpu_data();
                    for (int i = 0; i < mean_blob.channels(); ++i)
                    {
                        /* Extract an individual channel. */
                        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
                        channels.push_back(channel);
                        data += mean_blob.height() * mean_blob.width();
                    }

                    /* Merge the separate channels into a single image. */
                    cv::Mat mean;
                    cv::merge(channels, mean);
                    channel_mean = cv::mean(mean);
                }
            }
        }
    }
    if (NN == nullptr || weightsLoaded == false)
    {
        MO_LOG_EVERY_N(debug, 1000) << "Model not loaded";
        return false;
    }

    return true;
}

void CaffeImageClassifier::nodeInit(bool firstInit)
{
    (void)firstInit;
    aq::caffe_init_singleton::inst();
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}
template <class T1, class T2>
bool operator!=(const cv::Size_<T1>& lhs, const cv::Size_<T2>& rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

std::vector<std::vector<cv::cuda::GpuMat>> CaffeImageClassifier::getNetImageInput(int requested_batch_size)
{
    (void)requested_batch_size;
    std::vector<std::vector<cv::cuda::GpuMat>> output;
    auto data_itr = wrapped_inputs.find("data");
    if (data_itr == wrapped_inputs.end())
    {
        auto f = [this]() -> std::string {
            std::stringstream ss;
            for (auto& input : wrapped_inputs)
                ss << input.first;
            return ss.str();
        };
        MO_LOG(warning) << "Input blob \"data\" not found in network input blobs, existing blobs: " << f();
    }
    else
    {
        for (size_t i = 0; i < data_itr->second.size(); ++i)
        {
            output.push_back(data_itr->second[i].getGpuMatVec(stream()));
        }
    }
    return output;
}

cv::Scalar_<unsigned int> CaffeImageClassifier::getNetworkShape() const
{
    cv::Scalar_<unsigned int> output;
    auto data_itr = wrapped_inputs.find("data");
    if (data_itr != wrapped_inputs.end())
    {
        output[0] = data_itr->second.size();
        output[1] = data_itr->second[0].getChannels();
        auto sz = data_itr->second[0].getSize();
        output[2] = sz.height;
        output[3] = sz.width;
    }
    return output;
}

void CaffeImageClassifier::preBatch(int batch_size)
{
    if (!CheckInput())
        WrapInput();
    (void)batch_size;
    for (auto& handler : net_handlers)
    {
        handler->startBatch();
    }
}

void CaffeImageClassifier::postMiniBatch(const std::vector<cv::Rect>& batch_bb,
                                         const std::vector<DetectedObject2d>& dets)
{
    if (net_handlers.empty())
    {
        auto constructors = mo::MetaObjectFactory::instance()->getConstructors(Caffe::NetHandler::s_interfaceID);
        // For each blob, we check each handler and pick the handler with the highest priority
        std::map<int, std::vector<std::pair<int, IObjectConstructor*>>> blob_priority_map;
        for (auto& constructor : constructors)
        {
            auto info = dynamic_cast<Caffe::NetHandlerInfo*>(constructor->GetObjectInfo());
            if (info)
            {
                std::map<int, int> handled_blobs = info->CanHandleNetwork(*NN);
                for (auto& itr : handled_blobs)
                {
                    blob_priority_map[itr.first].emplace_back(itr.second, constructor);
                }
            }
        }
        for (auto& itr : blob_priority_map)
        {
            std::sort(itr.second.begin(),
                      itr.second.end(),
                      [](const std::pair<int, IObjectConstructor*>& I1, const std::pair<int, IObjectConstructor*>& I2) {
                          return I1.first > I2.first;
                      });
            if (itr.second.size() == 0)
            {
                continue;
            }
            // construct the handlers with largest priority
            auto obj = itr.second[0].second->Construct();
            auto handler = dynamic_cast<Caffe::NetHandler*>(obj);
            if (handler)
            {
                handler->Init(true);
                handler->setContext(this->getContext());
                handler->setLabels(&this->labels);
                net_handlers.emplace_back(handler);
                this->_algorithm_components.emplace_back(handler);
                handler->setOutputBlob(*NN, itr.first);
                handler->startBatch();
            }
            else
            {
                delete obj;
            }
        }
    }
    for (auto& handler : net_handlers)
    {
        handler->handleOutput(*NN, batch_bb, input_param, dets);
    }
}

void CaffeImageClassifier::postBatch()
{
    for (auto& handler : net_handlers)
    {
        handler->endBatch(input_param.getTimestamp());
    }
}

bool CaffeImageClassifier::forwardMinibatch()
{
    float loss;
    mo::scoped_profile profile_forward("Neural Net forward pass", nullptr, nullptr, cudaStream());
    NN->Forward(&loss);
    return true;
}

void CaffeImageClassifier::postSerializeInit()
{
    Node::postSerializeInit();
    for (auto& component : _algorithm_components)
    {
        rcc::shared_ptr<Caffe::NetHandler> handler(component);
        if (handler)
        {
            net_handlers.push_back(handler);
            handler->setLabels(&this->labels);
            handler->setContext(this->getContext());
        }
    }
}

MO_REGISTER_CLASS(CaffeImageClassifier)
