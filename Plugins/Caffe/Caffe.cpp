#define PARAMTERS_GENERATE_PERSISTENCE

#include "Caffe.h"
#include "caffe_init.h"

#include "EagleLib/nodes/Node.h"
#include "EagleLib/Nodes/NodeInfo.hpp"
#include <EagleLib/ObjectDetection.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudawarping.hpp>
#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
#include "caffe_include.h"

#include <boost/tokenizer.hpp>

#include <string>


#include "caffe/caffe.hpp"


#include "MetaObject/Logging/Log.hpp"

SETUP_PROJECT_IMPL;


using namespace EagleLib;
using namespace EagleLib::Nodes;

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


template <typename T>
std::vector<size_t> sort_indexes(const T* begin, size_t size) {

  // initialize original index locations
  std::vector<size_t> idx(size);
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), [&begin](size_t i1, size_t i2) {return begin[i1] < begin[i2];});

  return idx;
}
template <typename T>
std::vector<size_t> sort_indexes_ascending(const T* begin, size_t size) {

    // initialize original index locations
    std::vector<size_t> idx(size);
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&begin](size_t i1, size_t i2) {return begin[i1] > begin[i2]; });

    return idx;
}
template <typename T>
std::vector<size_t> sort_indexes(const T* begin, const T* end) {
    return sort_indexes<T>(begin, end - begin);
}
std::vector<SyncedMemory> CaffeBase::WrapBlob(caffe::Blob<float>& blob)
{
    std::vector<SyncedMemory> wrapped_blob;
    int height = blob.height();
    int width = blob.width();
    float* d_ptr = blob.mutable_gpu_data();
    float* h_ptr = blob.mutable_cpu_data();
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
        SyncedMemory image(h_wrappedChannels, d_wrappedChannels);
        wrapped_blob.push_back(image);
    }
    return wrapped_blob;
}
std::vector<SyncedMemory> CaffeBase::WrapBlob(caffe::Blob<double>& blob)
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
        SyncedMemory image(h_wrappedChannels, d_wrappedChannels);
        wrapped_blob.push_back(image);
    }
    return wrapped_blob;
}
void CaffeBase::WrapInput()
{
    if(NN == nullptr)
    {
        BOOST_LOG_TRIVIAL(error) << "Neural network not defined";
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
    LOG(debug) << ss.str();

    for(int k = 0; k < input_blobs.size(); ++k)
    {
        wrapped_inputs[input_names[k]] = WrapBlob(*input_blobs[k]);
    }
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
    for(int i = 0; i < output_idx.size(); ++i)
    {
        wrapped_outputs[NN->blob_names()[output_idx[i]]] = WrapBlob(*outputs[i]);
    }

    //caffe::Blob<float>* output_layer = NN->output_blobs()[0];
    //float* begin = output_layer->mutable_cpu_data();
    //float* end = begin + output_layer->channels()*output_layer->num();
    //wrapped_output = cv::Mat(1, end - begin, CV_32F, begin);
}
bool CaffeBase::InitNetwork()
{
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
    if (nn_model_file_param.modified)
    {
        if (boost::filesystem::exists(nn_model_file))
        {
            NN.reset(new caffe::Net<float>(nn_model_file.string(), caffe::TEST));
            WrapInput();
            nn_model_file_param.modified = false;
        }
        else
        {
            BOOST_LOG_TRIVIAL(debug) << "Architecture file does not exist";
        }
    }
    if (nn_weight_file_param.modified && NN)
    {
        if (boost::filesystem::exists(nn_weight_file))
        {
            NN->CopyTrainedLayersFrom(nn_weight_file.string());
            const std::vector<boost::shared_ptr<caffe::Layer<float>>>& layers = NN->layers();
            std::vector<std::string> layerNames;
            layerNames.reserve(layers.size());
            for (auto layer : layers)
            {
                layerNames.push_back(std::string(layer->type()));
            }
            BOOST_LOG_TRIVIAL(info) << "Weights loaded";
            weightsLoaded = true;
            UpdateParameter("Loaded layers", layerNames);
            nn_weight_file_param.modified = false;
        }
        else
        {
            BOOST_LOG_TRIVIAL(debug) << "Weight file does not exist";
        }
    }
    if (label_file_param.modified)
    {
        if (boost::filesystem::exists(label_file))
        {
            std::ifstream ifs(label_file.string().c_str());
            if (!ifs)
            {
                BOOST_LOG_TRIVIAL(error) << "Unable to load label file";
            }
            labels.reset(new std::vector<std::string>());
            std::string line;
            while (std::getline(ifs, line))
            {
                labels->push_back(line);
            }
            BOOST_LOG_TRIVIAL(info) << "Loaded " << labels->size() << " classes";
            label_file_param.modified = false;
        }
    }
    if (mean_file_param.modified)
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
        BOOST_LOG_TRIVIAL(trace) << "Model not loaded";
        return false;
    }
    return true;
}

void CaffeBase::NodeInit(bool firstInit)
{
    EagleLib::caffe_init_singleton::inst();
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}

bool CaffeImageClassifier::ProcessImpl()
{
    if(!InitNetwork())
        return false;
    cv::cuda::GpuMat float_image;
    if (input->GetDepth() != CV_32F)
    {
        input->GetGpuMat(Stream()).convertTo(float_image, CV_32F, Stream());
    }
    else
    {
        float_image = input->GetGpuMat(Stream());
    }
    cv::cuda::multiply(float_image, cv::Scalar::all(scale), float_image, 1.0, -1, Stream());
    std::vector<cv::Rect> defaultROI;
    defaultROI.push_back(cv::Rect(cv::Point(), input->GetSize()));
    if (bounding_boxes == nullptr)
    {
        bounding_boxes = &defaultROI;
    }
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
    
    if (bounding_boxes->size() > data_itr->second.size())
    {
        BOOST_LOG_TRIVIAL(debug) << "Too many input Regions of interest to handle in one pass, this network can only handle " << data_itr->second.size() << " inputs at a time";
    }
    auto shape = data_itr->second[0].GetShape();
    cv::Size input_size(shape[1], shape[2]);

    for (int i = 0; i < bounding_boxes->size() && i < data_itr->second.size(); ++i)
    {
        cv::cuda::GpuMat resized;
        if ((*bounding_boxes)[i].size() != input_size)
        {
            cv::cuda::resize(float_image, resized, input_size, 0, 0, cv::INTER_LINEAR, Stream());
        }
        else
        {
            resized = float_image((*bounding_boxes)[i]);
        }
        ;
        cv::cuda::split(resized, data_itr->second[i].GetGpuMatVecMutable(Stream()), Stream());
    }
    // Signal update on all inputs
    for(auto blob : input_blobs)
    {
        blob->mutable_gpu_data(); 
    }
    float loss;
    NN->ForwardPrefilled(&loss);
    caffe::Blob<float>* output_layer = NN->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels() * output_layer->num();
    const size_t step = output_layer->channels();
    
    
    std::vector<DetectedObject> objects(std::min<size_t>(bounding_boxes->size(), data_itr->second.size()));
    for (int i = 0; i < bounding_boxes->size() && i < data_itr->second.size(); ++i)
    {
        auto idx = sort_indexes_ascending(begin + i * output_layer->channels(), (size_t)output_layer->channels());
        objects[i].detections.resize(num_classifications);
        for (int j = 0; j < num_classifications; ++j)
        {
            objects[i].detections[j].confidence = (begin + i * output_layer->channels())[idx[j]];
            objects[i].detections[j].classNumber = idx[j];
            if (labels && idx[j] < labels->size())
            {
                objects[i].detections[j].label = (*labels)[idx[j]];
            }
        }
        objects[i].boundingBox = (*bounding_boxes)[i];
    }
    detections_param.UpdateData(objects, input_param.GetTimestamp(), _ctx);

    return true;
}

MO_REGISTER_CLASS(CaffeImageClassifier)
