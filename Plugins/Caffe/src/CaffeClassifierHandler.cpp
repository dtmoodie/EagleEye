#include "CaffeClassifierHandler.hpp"
#include "CaffeNetHandlerInfo.hpp"
using namespace aq::Caffe;

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


std::map<int, int> ClassifierHandler::CanHandleNetwork(const caffe::Net<float>& net)
{
    const std::vector<int>& out_idx = net.output_blob_indices();
    auto layer_names = net.layer_names();
    auto layers = net.layers();
    std::map<int, int> output;
    for(int i = 0; i < layer_names.size(); ++i)
    {
        std::vector<int> top_ids = net.top_ids(i);
        for(auto id : top_ids)
        {
            if(std::find(out_idx.begin(), out_idx.end(), id) != out_idx.end())
            {
                // Layer(i) outputs from network
                std::string type = layers[i]->type();
                if(type == "Softmax" || type == "Convolution" || type == "Crop")
                {
                    const std::vector<boost::shared_ptr<caffe::Blob<float> > >& blobs = net.blobs();
                    const std::vector<int>& shape = blobs[id]->shape();
                    if(shape.size() == 4)
                    {
                        if(shape[2] == 1 && shape[3] == 1)
                            output[id] = 10;
                    }
                    if(shape.size() == 2)
                        output[id] = 10;
                }
            }
        }
    }
    return output;
}
void ClassifierHandler::StartBatch()
{
    objects.clear();
}

void ClassifierHandler::HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size)
{
    auto output_blob = net.blob_by_name(output_blob_name);
    if(output_blob)
    {
        float* data = output_blob->mutable_cpu_data();
        int num = output_blob->channels();
        //objects.resize(output_blob->num());

        for(int i = 0; i  < output_blob->num() && i < bounding_boxes.size(); ++i)
        {
            auto idx = sort_indexes_ascending(data + i * num, num);
            DetectedObject obj;
            obj.timestamp = timestamp;
            if(labels && idx[0] < labels->size())
            {
                obj.detections.emplace_back((*labels)[idx[0]], (data + i * num)[idx[0]], idx[0]);
            }else
            {
                obj.detections.emplace_back("", (data + i * num)[idx[0]], idx[0]);
            }
            obj.boundingBox = cv::Rect2f(bounding_boxes[i].x, bounding_boxes[i].y, bounding_boxes[i].width, bounding_boxes[i].height);
            objects.push_back(obj);
        }
    }
}
void ClassifierHandler::EndBatch(long long timestamp)
{
    objects_param.Commit(timestamp, _ctx);
}

MO_REGISTER_CLASS(ClassifierHandler)
