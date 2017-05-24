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

void ClassifierHandler::HandleOutput(const caffe::Net<float>& net, const std::vector<cv::Rect>& bounding_boxes, mo::ITypedParameter<aq::SyncedMemory>& input_param, const std::vector<DetectedObject2d>& objs){
    auto output_blob = net.blob_by_name(output_blob_name);
    if(output_blob)
    {
        float* data = output_blob->mutable_cpu_data();
        int num = output_blob->channels();

        for(int i = 0; i  < output_blob->num() && i < bounding_boxes.size(); ++i)
        {
            auto idx = sort_indexes_ascending(data + i * num, num);
            DetectedObject obj;
            obj.timestamp = input_param.getTimestamp();
            if(labels && idx[0] < labels->size())
            {
                obj.classification = Classification((*labels)[idx[0]], (data + i * num)[idx[0]], idx[0]);
            }else
            {
                obj.classification =  Classification("", (data + i * num)[idx[0]], idx[0]);
            }
            obj.boundingBox = cv::Rect2f(bounding_boxes[i].x, bounding_boxes[i].y, bounding_boxes[i].width, bounding_boxes[i].height);
            if(objs.size() == bounding_boxes.size())
            {
                obj.id = objs[i].id;
                obj.framenumber = objs[i].framenumber;
                obj.timestamp = objs[i].timestamp;
                obj.boundingBox = objs[i].boundingBox;
            }
            objects.push_back(obj);
        }
    }
}

void ClassifierHandler::EndBatch(boost::optional<mo::Time_t>timestamp)
{
    objects_param.emitUpdate(timestamp, _ctx);
}

MO_REGISTER_CLASS(ClassifierHandler)
