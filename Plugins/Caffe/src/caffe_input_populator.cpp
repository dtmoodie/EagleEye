#include "caffe_input_populator.h"
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>
#include <caffe/blob.hpp>
using namespace aq;
using namespace aq::nodes;

/*void caffe_input_populator::fill_blobs()
{
    auto fb = getParameter<std::vector<TS<SyncedMemory>>>("frame cache")->Data();
    auto roi = getParameter<std::map<int, std::vector<std::pair<cv::Rect, int>>>>("regions of interest")->Data();
    auto input = getParameter<std::vector<std::vector<std::vector<cv::Mat>>>>("network input blobs")->Data();
    auto class_map = getParameter<std::map<int, int>>("class map")->Data();
    if(fb && input)
    {
        if(sample_permutation.empty())
        {
            // Assumes all data is loaded when you call this
            for(auto& img : *fb)
            {
                if(roi)
                {
                    auto itr = roi->find(img.frame_number);
                    if(itr != roi->end())
                    {
                        for(int i = 0; i < itr->second.size(); ++i)
                        {
                            sample_permutation.push_back(std::make_pair((int)img.frame_number, i));
                        }
                    }
                }else if(class_map)
                {
                    if(class_map->find(img.frame_number) != class_map->end())
                    {
                        sample_permutation.push_back(std::make_pair(img.frame_number, -1));
                    }
                }
            }
            if(shuffle)
            {
                std::random_shuffle(sample_permutation.begin(), sample_permutation.end());
            }
        }
        if(blob_index < input->size() && blob_index >= 0)
        {
            auto& blob = (*input)[blob_index];
            int num = blob.size();
            CV_Assert(blob.size() && "Requires Samples dimension");
            CV_Assert(blob[0].size() && "Requires channels");
            int channels = blob[0].size();
            int width = blob[0][0].cols;
            int height = blob[0][0].rows;
            int count = 0;
            for(int i = sample_index; i < sample_index + num && i < sample_permutation.size(); ++i, ++count)
            {
                auto& permutation = sample_permutation[i];

                cv::Mat mat; // = (*fb)[permutation.first].getMat();
                if(permutation.second != -1)
                {
                    mat =
(*fb)[permutation.first].getMat(cv::cuda::Stream::Null())((*roi)[permutation.first][permutation.second].first);
                }else
                {
                    mat = (*fb)[permutation.first].getMat(cv::cuda::Stream::Null());
                }
                // Check if we need to resize or do channel stuff
                if(mat.size() != cv::Size(width, height))
                {
                    cv::resize(mat, mat, cv::Size(width,height), 0, 0, cv::INTER_CUBIC);
                }
            }
        }
    }
}
void caffe_input_populator::nodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParam<std::vector<TS<SyncedMemory>>>("frame cache");
        addInputParam<std::map<int, std::vector<std::pair<cv::Rect, int>>>>("regions of interest");
        addInputParam<std::map<int, int>>("class map");
        addInputParam<std::vector<std::vector<std::vector<cv::Mat>>>>("network input blobs");
        addInputParam<std::vector<std::string>>("input blob names");
    }
    auto f = [this](cv::cuda::Stream* stream)
    {
        auto input = getParameter<std::vector<std::vector<std::vector<cv::Mat>>>>("network input blobs")->Data();
        auto names = getParameter<std::vector<std::string>>("input blob names")->Data();
        if( input && names && input->size() == names->size())
        {
            if(this->blob_index >= 0 && this->blob_index < input->size())
            {
                blob_name = (*names)[blob_index];
                blob_name_param.OnUpdate(stream);
            }
        }
    };
    RegisterParameterCallback("blob_index", f, true, true);
}

TS<SyncedMemory> caffe_input_populator::doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    return input;
}

bool caffe_input_populator::pre_check(const TS<SyncedMemory>& input)
{
    return true;
}

static aq::nodes::NodeInfo g_registerer_caffe_input_populator("caffe_input_populator", { "caffe"});
REGISTERCLASS(caffe_input_populator, &g_registerer_caffe_input_populator);*/