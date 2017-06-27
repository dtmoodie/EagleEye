#define OPENCV
#include "yolo_v2_class.hpp"

#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <INeuralNet.hpp>
#include <Aquila/nodes/IDetector.hpp>
#include <boost/filesystem.hpp>
class YOLO: virtual public aq::nodes::INeuralNet, virtual public aq::nodes::IImageDetector {
public:
    typedef aq::nodes::NodeInfo InterfaceInfo;
    static const unsigned int s_interfaceID = aq::nodes::Node::s_interfaceID;
    MO_DERIVE(YOLO, aq::nodes::INeuralNet, aq::nodes::IImageDetector)
    MO_END;
protected:
    virtual std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int batch_size = 1){
        if (!detector) {
            if(boost::filesystem::exists(model_file) && boost::filesystem::exists(weight_file)){
                detector = std::make_shared<Detector>(model_file.string(), weight_file.string());
            }else{
                LOG(warning) << "Model file '" << model_file << "' or weight file '" << weight_file << "' does not exist";
            }
        }
        if (input_buffer.empty() && detector) {
            int width, height;
            if (detector->get_network_size(width, height)) {
                workspace = cv::cuda::GpuMat(1, width * height * 3, CV_32F);
                float* ptr = workspace.ptr<float>();
                std::vector<cv::cuda::GpuMat> channels;
                for (int i = 0; i < 3; ++i) {
                    channels.emplace_back(height, width, CV_32F, static_cast<void*>(ptr), width);
                    ptr += height * width;
                }
                if (swap_bgr) {
                    std::swap(channels[0], channels[2]);
                }
                input_buffer.push_back(channels);
                input_ptr = ptr;
            }
        }
        return input_buffer;
    }
    virtual bool forwardMinibatch(){
        if(input_ptr){
            current_detections = detector->detect(input_ptr);
            return true;
        }
        return false;
    }
    virtual void postBatch(const std::vector<cv::Rect>& batch_bb, const std::vector<aq::DetectedObject2d>& dets){
        // do things with current_detections
    }
    virtual bool reshapeNetwork(int num, int channels, int height, int width){
        return false;
    }
private:
    std::shared_ptr<Detector> detector;
    std::vector<std::vector<cv::cuda::GpuMat>> input_buffer;
    cv::cuda::GpuMat workspace;
    float* input_ptr = nullptr;
    std::vector<bbox_t> current_detections;
};

MO_REGISTER_CLASS(YOLO)