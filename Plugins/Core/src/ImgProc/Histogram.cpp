#include "Histogram.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
using namespace aq::Nodes;

bool HistogramRange::processImpl()
{
    if(lower_bound_param.modified() || upper_bound_param.modified() || bins_param.modified())
    {
        updateLevels(input->getDepth());
        lower_bound_param.modified(false);
        upper_bound_param.modified(false);
        bins_param.modified(false);
    }
    if(input->getChannels() == 1 || input->getChannels() == 4)
    {
        cv::cuda::GpuMat hist;
        cv::cuda::histRange(input->getGpuMat(stream()), hist, levels.getGpuMat(stream()), stream());
        histogram_param.updateData(hist, input_param.getTimestamp(), _ctx);
        return true;
    }
    return false;
}


void HistogramRange::updateLevels(int type)
{
    cv::Mat h_mat;
    if(type == CV_32F)
        h_mat = cv::Mat(1, bins, CV_32F);
    else
        h_mat = cv::Mat(1, bins, CV_32S);
    double step = (upper_bound - lower_bound) / double(bins);

    double val = lower_bound;
    for(int i = 0; i < bins; ++i, val += step)
    {
        if(type == CV_32F)
            h_mat.at<float>(i) = val;
        if(type == CV_8U)
            h_mat.at<int>(i) = val;
    }
    levels_param.updateData(h_mat);
}

bool Histogram::processImpl()
{
    cv::cuda::GpuMat bins, hist;
    cv::cuda::histogram(input->getGpuMat(stream()), bins, hist, min, max, stream());
    histogram_param.updateData(hist, input_param.getTimestamp(), _ctx);
    return true;
}

MO_REGISTER_CLASS(HistogramRange)
MO_REGISTER_CLASS(Histogram)
