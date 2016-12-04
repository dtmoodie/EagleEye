#define PARAMTERS_GENERATE_PERSISTENCE
#include "Caffe.h"
#include "caffe_init.h"

#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/NodeInfo.hpp"
#include <EagleLib/ObjectDetection.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudawarping.hpp>

#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
#include <MetaObject/Logging/Profiling.hpp>
#include "MetaObject/Logging/Log.hpp"

#include "caffe_include.h"
#include <boost/tokenizer.hpp>

#include <string>

#include "caffe/caffe.hpp"

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
    for(int i = 0; i < input_blob_indecies.size(); ++i)
    {
        auto itr = wrapped_inputs.find(input_names[i]);
        if(itr != wrapped_inputs.end())
        {
            const float* data = input_blobs[i]->gpu_data();
            for(int j = 0; j < itr->second.size(); ++j)
            {
                for(int k = 0; k < itr->second[j].GetNumMats(); ++k)
                {
                    const cv::cuda::GpuMat& mat = itr->second[j].GetGpuMat(Stream(), k);
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
    for(int i = 0; i < output_idx.size(); ++i)
    {
        wrapped_outputs[NN->blob_names()[output_idx[i]]] = WrapBlob(*outputs[i]);
    }
    if(NN->has_layer("detection_out"))
    {
        _network_type = Detector_e;
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
    if(!has_fully_connected)
    {
        _network_type = (NetworkType)(_network_type | FCN_e);
    }
}

bool CaffeBase::InitNetwork()
{
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
    if (nn_model_file_param.modified)
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
            nn_model_file_param.modified = false;
        }
        else
        {
            LOG_EVERY_N(warning, 100) << "Architecture file does not exist " << nn_model_file.string();
        }
    }

    if (nn_weight_file_param.modified && NN)
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
            UpdateParameter("Loaded layers", layerNames);
            nn_weight_file_param.modified = false;
        }
        else
        {
            LOG_EVERY_N(warning, 100) << "Weight file does not exist " << nn_weight_file.string();
        }
    }

    if (label_file_param.modified)
    {
        if (boost::filesystem::exists(label_file))
        {
            std::ifstream ifs(label_file.string().c_str());
            if (!ifs)
            {
                LOG_EVERY_N(warning, 100) << "Unable to load label file";
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
        LOG_EVERY_N(debug, 1000) << "Model not loaded";
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

    if (input->empty())
        return false;

    if(!CheckInput())
        WrapInput();
    auto input_shape = input->GetShape();

    if(image_scale != -1)
        ReshapeInput(input_shape[0], input_shape[3], input_shape[1] * image_scale, input_shape[2]*image_scale);

    cv::cuda::GpuMat float_image;
    
    if (input->GetDepth() != CV_32F)
    {
        input->GetGpuMat(Stream()).convertTo(float_image, CV_32F, Stream());
    }
    else
    {
        input->GetGpuMat(Stream()).copyTo(float_image, Stream());
    }

    cv::cuda::subtract(float_image, channel_mean, float_image, cv::noArray(), -1, Stream());
    cv::cuda::multiply(float_image, cv::Scalar::all(pixel_scale), float_image, 1.0, -1, Stream());

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
    cv::Size input_size(shape[2], shape[1]);
    std::vector<cv::Mat> debug_mat_vec;
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
        cv::cuda::split(resized, data_itr->second[i].GetGpuMatVecMutable(Stream()), Stream());   
    }
    
    // Signal update on all inputs
    float* data = nullptr;
    for(auto blob : input_blobs)
    {
        data = blob->mutable_gpu_data();
    }

    
    if(debug_dump)
    {
        for (int i = 0; i < data_itr->second[0].GetNumMats(); ++i)
        {
            debug_mat_vec.push_back(cv::Mat(data_itr->second[0].GetGpuMat(Stream(), i)));
        }
    }

    float loss;
    {
        mo::scoped_profile profile("Neural Net forward pass", &_rmt_hash, &_rmt_cuda_hash, &Stream());
        NN->Forward(&loss);
    }
    
    caffe::Blob<float>* output_layer = NN->output_blobs()[0];
    float* begin = output_layer->mutable_cpu_data();
    const float* end = begin + output_layer->channels() * output_layer->num();
    const size_t step = output_layer->channels();
    
    if(_network_type & Classifier_e)
    {
        std::vector<DetectedObject> objects(std::min<size_t>(bounding_boxes->size(), data_itr->second.size()));
        for (int i = 0; i < bounding_boxes->size() && i < data_itr->second.size(); ++i)
        {
            auto idx = sort_indexes_ascending(begin + i * output_layer->channels(), (size_t)output_layer->channels());
            objects[i].detections.resize(num_classifications);
            for (int j = 0; j < num_classifications && j < idx.size(); ++j)
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
    }else if(_network_type & Detector_e)
    {
        const int num_detections = output_layer->height();
        cv::Mat all(num_detections, 7, CV_32F, begin);
        cv::Mat_<float> labels(num_detections, 1, begin + 1, sizeof(float)*7);
        cv::Mat_<float> confidence(num_detections, 1, begin + 2, sizeof(float)*7);
        cv::Mat_<float> xmin(num_detections, 1, begin + 3, sizeof(float) * 7);
        cv::Mat_<float> ymin(num_detections, 1, begin + 4, sizeof(float) * 7);
        cv::Mat_<float> xmax(num_detections, 1, begin + 5, sizeof(float) * 7);
        cv::Mat_<float> ymax(num_detections, 1, begin + 6, sizeof(float) * 7);
        std::vector<DetectedObject> objects;
        cv::Size original_size = input->GetSize();
        for(int i = 0; i < num_detections; ++i)
        {
            if(confidence[i][0] > detection_threshold)
            {
                DetectedObject obj;
                obj.boundingBox.x = xmin[i][0] * original_size.width;
                obj.boundingBox.y = ymin[i][0] * original_size.height;
                obj.boundingBox.width = (xmax[i][0] - xmin[i][0])*original_size.width;
                obj.boundingBox.height = (ymax[i][0] - ymin[i][0]) * original_size.height;
                if (this->labels && labels[i][0] < this->labels->size())
                    obj.detections.emplace_back((*this->labels)[int(labels[i][0])], confidence[i][0], int(labels[i][0]));
                else
                    obj.detections.emplace_back("", confidence[i][0], int(labels[i][0]));

                objects.push_back(obj);
            }
            if(objects.size())
            {
                LOG(trace) << "Detected " << objects.size() << " objets in frame " << input_param.GetTimestamp();
            }
        }
        if(!(objects.empty() && detections.empty()))
            detections_param.UpdateData(objects, input_param.GetTimestamp(), _ctx);
    }
    
    bounding_boxes = nullptr;
    return true;
}

MO_REGISTER_CLASS(CaffeImageClassifier)
