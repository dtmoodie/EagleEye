#define PARAMTERS_GENERATE_PERSISTENCE
#include "Caffe.h"
#include "caffe_init.h"
#include "helpers.hpp"
#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>
#include "helpers.hpp"
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/Types.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <MetaObject/logging/Profiling.hpp>
#include "MetaObject/logging/Log.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include "caffe_include.h"
#include <boost/tokenizer.hpp>

#include <string>

#include "caffe/caffe.hpp"

using namespace aq;
using namespace aq::nodes;
#ifndef _MSC_VER
#include "dlfcn.h"
#else

#endif
void InitModule()
{
#ifndef _MSC_VER
    dlopen("libpython2.7.so", RTLD_LAZY | RTLD_GLOBAL);
#endif
}


std::vector<SyncedMemory> CaffeBase::WrapBlob(caffe::Blob<float>& blob, bool bgr_swap)
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
            d_ptr += height*width;
            h_ptr += height*width;
        }
        if(bgr_swap && h_wrappedChannels.size() == 3 && d_wrappedChannels.size() == 3)
        {
            std::swap(h_wrappedChannels[0], h_wrappedChannels[2]);
            std::swap(d_wrappedChannels[0], d_wrappedChannels[2]);
        }
        SyncedMemory image(h_wrappedChannels, d_wrappedChannels, SyncedMemory::DO_NOT_SYNC);
        wrapped_blob.push_back(image);
    }
    return wrapped_blob;
}

std::vector<SyncedMemory> CaffeBase::WrapBlob(caffe::Blob<double>& blob, bool bgr_swap)
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
            d_ptr += height*width;
            h_ptr += height*width;
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

void CaffeBase::WrapInput()
{
    if(NN == nullptr)
    {
        LOG_EVERY_N(error, 100) << "Neural network not defined";
        return;
    }
    if(NN->num_inputs() == 0)
        return;
    auto input_blob_indecies = NN->input_blob_indices();
    std::vector<std::string> input_names;
    for(auto idx : input_blob_indecies)
    {
        input_names.push_back(NN->blob_names()[idx]);
    }
    input_blobs = NN->input_blobs();

    std::stringstream ss;
    ss << "Architecture loaded, num inputs: " << NN->num_inputs();
    ss << " num outputs: " << NN->num_outputs() << "\n";
    for(int i = 0; i < input_blobs.size(); ++i)
    {
        ss << "   input batch size: " << input_blobs[i]->num() << "\n";
        ss << "   input channels: " << input_blobs[i]->channels() << "\n";
        ss << "   input size: (" << input_blobs[i]->width() << ", " << input_blobs[i]->height() << ")\n";
    }
    //LOG(debug) << ss.str();

    for(int k = 0; k < input_blobs.size(); ++k)
    {
        wrapped_inputs[input_names[k]] = WrapBlob(*input_blobs[k], bgr_swap);
    }
}

bool CaffeBase::CheckInput()
{
    if(NN == nullptr)
        return false;
    const auto& input_blob_indecies = NN->input_blob_indices();
    std::vector<std::string> input_names;
    for (auto idx : input_blob_indecies)
    {
        input_names.push_back(NN->blob_names()[idx]);
    }
    auto input_blobs_ = NN->input_blobs();
    if(input_blobs_.size() != input_blobs.size())
        return false;
    for(int i = 0; i < input_blob_indecies.size(); ++i)
    {
        auto itr = wrapped_inputs.find(input_names[i]);
        if(itr != wrapped_inputs.end())
        {
            const float* data = input_blobs[i]->gpu_data();
            for(int j = 0; j < itr->second.size(); ++j)
            {
                for(int k = 0; k < itr->second[j].getNumMats(); ++k)
                {
                    const cv::cuda::GpuMat& mat = itr->second[j].getGpuMat(stream(), k);
                    if(data != (float*)mat.data)
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

void CaffeBase::ReshapeInput(int num, int channels, int height, int width)
{
    input_blobs = NN->input_blobs();
    for(auto input_blob : input_blobs)
    {
        input_blob->Reshape(num, channels, height, width);
    }
    if(!CheckInput())
        WrapInput();
}

void CaffeBase::WrapOutput()
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
    for(int i = 0; i < output_idx.size(); ++i){
        wrapped_outputs[NN->blob_names()[output_idx[i]]] = WrapBlob(*outputs[i]);
    }
    auto layers = NN->layers();
    bool has_fully_connected = false;
    for(auto layer : layers)
    {
        if(layer->type() == std::string("InnerProduct"))
        {
            has_fully_connected = true;
        }
    }
}

bool CaffeBase::InitNetwork()
{
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
    if (nn_model_file_param.modified())
    {
        if (boost::filesystem::exists(nn_model_file))
        {
            std::string param_file = nn_model_file.string();
            try
            {
                NN.reset(new caffe::Net<float>(param_file, caffe::TEST));
            }catch(caffe::ExceptionWithCallStack<std::string>& exp)
            {
                throw mo::ExceptionWithCallStack<std::string>(exp, exp.CallStack());
            }
            WrapInput();
            WrapOutput();
            nn_model_file_param.modified(false);
        }
        else
        {
            LOG_EVERY_N(warning, 100) << "Architecture file does not exist " << nn_model_file.string();
        }
    }

    if (nn_weight_file_param.modified() && NN)
    {
        if (boost::filesystem::exists(nn_weight_file))
        {
            try
            {
                NN->CopyTrainedLayersFrom(nn_weight_file.string());
            }
            catch (caffe::ExceptionWithCallStack<std::string>& exp)
            {
                throw mo::ExceptionWithCallStack<std::string>(exp, exp.CallStack());
            }catch (...)
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
            //UpdateParameter("Loaded layers", layerNames);
            nn_weight_file_param.modified(false);
        }
        else
        {
            LOG_EVERY_N(warning, 100) << "Weight file does not exist " << nn_weight_file.string();
        }
    }

    if ((label_file_param.modified() || labels.empty()) && boost::filesystem::exists(label_file))
    {
        labels.clear();
        std::ifstream ifs(label_file.string().c_str());
        if (!ifs)
        {
            LOG_EVERY_N(warning, 100) << "Unable to load label file";
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
        LOG_EVERY_N(debug, 1000) << "Model not loaded";
        return false;
    }
    return true;
}

void CaffeBase::nodeInit(bool firstInit)
{
    aq::caffe_init_singleton::inst();
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}
template<class T1, class T2>
bool operator !=(const cv::Size_<T1>& lhs, const cv::Size_<T2>& rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

bool CaffeImageClassifier::processImpl()
{
     if(!InitNetwork())
        return false;

    if (input->empty())
        return false;

    if(!CheckInput())
        WrapInput();
    auto input_shape = input->getShape();
    if(input_blobs.empty())
    {
        LOG_EVERY_N(warning, 100) << "No input blobs to network, is this a deploy network?";
        return false;
    }
    if(input_shape[3] != input_blobs[0]->channels())
    {
        LOG(warning) << "Cannot handle image with " << input_shape[3] << " channels with network "
                     << " designed for " << input_blobs[0]->channels() << " channels";
        return false;
    }

    std::vector<cv::Rect2f> defaultROI;
    defaultROI.push_back(cv::Rect2f(0,0, 1.0, 1.0));

    if (bounding_boxes == nullptr)
    {
        bounding_boxes = &defaultROI;
    }

    if(input_detections != nullptr && bounding_boxes == &defaultROI)
    {
        defaultROI.clear();
        for(const auto& itr : *input_detections)
        {
            defaultROI.emplace_back(
                    itr.boundingBox.x / input_shape[2],
                    itr.boundingBox.y / input_shape[1],
                    itr.boundingBox.width / input_shape[2],
                    itr.boundingBox.height / input_shape[1]);
        }
        if(defaultROI.size() == 0)
        {
            for(auto& handler : net_handlers)
            {
                handler->startBatch();
                handler->endBatch(input_param.getTimestamp());
            }
            if(bounding_boxes == &defaultROI)
            {
                bounding_boxes = nullptr;
            }
            return false;
        }
    }

    std::vector<cv::Rect> pixel_bounding_boxes;
    for(int i = 0; i < bounding_boxes->size(); ++i)
    {
        cv::Rect bb;
        bb.x = (*bounding_boxes)[i].x * input_shape[2];
        bb.y = (*bounding_boxes)[i].y * input_shape[1];
        bb.width = (*bounding_boxes)[i].width * input_shape[2];
        bb.height = (*bounding_boxes)[i].height * input_shape[1];
        if(bb.x + bb.width >= input_shape[2])
        {
            bb.x -= input_shape[2] - bb.width;
        }
        if(bb.y + bb.height >= input_shape[1])
        {
            bb.y -= input_shape[1] - bb.height;
        }
        bb.x = std::max(0, bb.x);
        bb.y = std::max(0, bb.y);
        pixel_bounding_boxes.push_back(bb);
    }


    if(image_scale != -1)
    {
        ReshapeInput(bounding_boxes->size(), input_shape[3], input_shape[1] * image_scale, input_shape[2]*image_scale);
    }

    cv::cuda::GpuMat float_image;

    if (input->getDepth() != CV_32F)
    {
        input->getGpuMat(stream()).convertTo(float_image, CV_32F, stream());
    }
    else
    {
        input->getGpuMat(stream()).copyTo(float_image, stream());
    }

    cv::cuda::subtract(float_image, channel_mean, float_image, cv::noArray(), -1, stream());
    cv::cuda::multiply(float_image, cv::Scalar::all(pixel_scale), float_image, 1.0, -1, stream());

    auto data_itr = wrapped_inputs.find("data");
    if(data_itr == wrapped_inputs.end())
    {
        auto f = [this]()->std::string {
            std::stringstream ss;
            for (auto& input : wrapped_inputs)
                ss << input.first;
            return ss.str();
        };
        LOG(warning) << "Input blob \"data\" not found in network input blobs, existing blobs: " << f();

        return false;
    }
    auto shape = data_itr->second[0].getShape();
    cv::Size input_size(shape[2], shape[1]); // NN input size

    if(pixel_bounding_boxes.size() != input_shape[0] &&
            input_detections == nullptr /* Since number of detections can change at each iteration, it is beneficial to avoid reshaping the minibatch size constantly */)
    {
        ReshapeInput(pixel_bounding_boxes.size(), input_shape[3], shape[1], shape[2]);
    }

    cv::cuda::GpuMat resized;
    for(auto& handler : net_handlers)
    {
        handler->startBatch();
    }
    cv::Size input_image_size = input->getSize();
    for(int i = 0; i < pixel_bounding_boxes.size();) // for each roi
    {
        int start = i, end = 0;
        for(int j = 0; j < data_itr->second.size() && i < pixel_bounding_boxes.size(); ++j, ++i) // for each mini batch
        {
            if (pixel_bounding_boxes[i].size() != input_size){
                cv::cuda::resize(float_image(pixel_bounding_boxes[i]), resized, input_size, 0, 0, cv::INTER_LINEAR, stream());
            }else{
                resized = float_image(pixel_bounding_boxes[i]);
            }
            cv::cuda::split(resized, data_itr->second[j].getGpuMatVecMutable(stream()), stream());
            end = start + j + 1;
        }
        // Signal update on all inputs
        for(auto blob : input_blobs){
            blob->mutable_gpu_data();
        }

        float loss;
        {
            mo::scoped_profile profile_forward("Neural Net forward pass", nullptr, nullptr, cudaStream());
            NN->Forward(&loss);
        }

        if(net_handlers.empty())
        {
            auto constructors = mo::MetaObjectFactory::instance()->getConstructors(Caffe::NetHandler::s_interfaceID);
            // For each blob, we check each handler and pick the handler with the highest priority
            std::map<int, std::vector<std::pair<int, IObjectConstructor*>>> blob_priority_map;
            for(auto& constructor : constructors)
            {
                auto info = dynamic_cast<Caffe::NetHandlerInfo*>(constructor->GetObjectInfo());
                if(info)
                {
                    std::map<int,int> handled_blobs = info->CanHandleNetwork(*NN);
                    for(auto& itr : handled_blobs)
                    {
                        blob_priority_map[itr.first].emplace_back(itr.second, constructor);
                    }
                }
            }
            for(auto& itr : blob_priority_map)
            {
                std::sort(itr.second.begin(), itr.second.end(), [](const std::pair<int, IObjectConstructor*>& I1,const std::pair<int, IObjectConstructor*>& I2)
                {
                   return I1.first > I2.first;
                });
                if(itr.second.size() == 0)
                {
                    continue;
                }
                auto obj = itr.second[0].second->Construct();
                auto handler = dynamic_cast<Caffe::NetHandler*>(obj);
                if(handler)
                {
                    handler->Init(true);
                    handler->setContext(this->getContext());
                    handler->setLabels(&this->labels);
                    net_handlers.emplace_back(handler);
                    this->_algorithm_components.emplace_back(handler);
                    handler->setOutputBlob(*NN, itr.first);
                    handler->startBatch();

                }else
                {
                    delete obj;
                }
                // construct the handlers with largest priority
            }
        }
        mo::scoped_profile profile_handlers("Handle neural net output", nullptr, nullptr, cudaStream());
        std::vector<cv::Rect> batch_bounding_boxes;
        std::vector<DetectedObject2d> batch_objects;
        for(int j = start; j < end; ++j){
            batch_bounding_boxes.push_back(pixel_bounding_boxes[j]);
        }
        if(input_detections != nullptr && bounding_boxes == &defaultROI){
            for(int j = start; j < end; ++j)
                batch_objects.push_back((*input_detections)[j]);
        }
        for(auto& handler : net_handlers){
            handler->handleOutput(*NN, batch_bounding_boxes, input_param, batch_objects);
        }
    }
    for(auto& handler : net_handlers)
    {
        handler->endBatch(input_param.getTimestamp());
    }
    if(bounding_boxes == &defaultROI)
    {
        bounding_boxes = nullptr;
    }
    return true;
}
void CaffeImageClassifier::postSerializeInit()
{
    Node::postSerializeInit();
    for(auto& component : _algorithm_components)
    {
        rcc::shared_ptr<Caffe::NetHandler> handler(component);
        if(handler)
        {
            net_handlers.push_back(handler);
            handler->setLabels(&this->labels);
            handler->setContext(this->getContext());
        }
    }
}

MO_REGISTER_CLASS(CaffeImageClassifier)
