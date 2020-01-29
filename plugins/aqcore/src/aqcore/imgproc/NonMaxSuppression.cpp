#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "NonMaxSuppression.h"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>

using namespace aq;
using namespace aq::nodes;

bool MinMax::processImpl()
{
    cv::cuda::minMax(input->getGpuMat(stream()), &min_value, &max_value);
    min_value_param.emitUpdate(input_param.getTimestamp(), _ctx.get());
    max_value_param.emitUpdate(input_param.getTimestamp(), _ctx.get());
    return true;
}

bool Threshold::processImpl()
{
    if (input_max)
        max_param.updateData(*input_max * input_percent);
    if (input_min)
        min_param.updateData(*input_min * input_percent);
    cv::cuda::GpuMat upper_mask, lower_mask;
    if (two_sided)
    {
        if (source_value)
        {
            cv::cuda::threshold(input->getGpuMat(stream()), upper_mask, max, replace_value, inverse ? 3 : 4, stream());
        }
        else
        {
            cv::cuda::threshold(input->getGpuMat(stream()), upper_mask, max, replace_value, inverse ? 1 : 0, stream());
        }
    }
    if (truncate)
    {
        cv::cuda::threshold(input->getGpuMat(stream()), lower_mask, min, replace_value, 2, stream());
    }
    else
    {
        if (source_value)
        {
            cv::cuda::threshold(input->getGpuMat(stream()), lower_mask, min, 0.0, inverse ? 4 : 3, stream());
        }
        else
        {
            cv::cuda::threshold(input->getGpuMat(stream()), lower_mask, min, replace_value, inverse ? 1 : 0, stream());
        }
    }
    cv::cuda::GpuMat mask;
    if (upper_mask.empty())
    {
        mask = lower_mask;
        // mask_param.updateData(lower_mask, input_param.getTimestamp(), _ctx.get());
    }
    else
    {
        cv::cuda::bitwise_and(upper_mask, lower_mask, mask, cv::noArray(), stream());
    }
    mask.convertTo(mask, input->getType(), stream());
    mask_param.updateData(mask, input_param.getTimestamp(), _ctx.get());
    return true;
}

/*cv::cuda::GpuMat NonMaxSuppression::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    // initialise the block mask and destination
    const int M = img.rows;
    const int N = img.cols;
    cv::Mat src(img);
    int sz = *getParameter<int>("Size")->Data();

    cv::Mat mask;
    if(auto maskParam = getParameter<cv::cuda::GpuMat>(1)->Data())
        if(!maskParam->empty())
            maskParam->download(mask, stream);
    stream.waitForCompletion();
    const bool masked = !mask.empty();
    cv::Mat block = 255*cv::Mat_<uint8_t>::ones(cv::Size(2*sz+1,2*sz+1));
    cv::Mat dst = cv::Mat_<uint8_t>::zeros(src.size());

    // iterate over image blocks
    for (int m = 0; m < M; m+=sz+1) {
        for (int n = 0; n < N; n+=sz+1) {
            cv::Point  ijmax;
            double vcmax, vnmax;

            // get the maximal candidate within the block
            cv::Range ic(m, cv::min(m+sz+1,M));
            cv::Range jc(n, cv::min(n+sz+1,N));

            cv::minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic,jc) : cv::noArray());


            cv::Point cc = ijmax + cv::Point(jc.start,ic.start);

            // search the neighbours centered around the candidate for the true maxima
            cv::Range in(cv::max(cc.y-sz,0), cv::min(cc.y+sz+1,M));
            cv::Range jn(cv::max(cc.x-sz,0), cv::min(cc.x+sz+1,N));

            // mask out the block whose maxima we already know
            cv::Mat_<uint8_t> blockmask;
            block(cv::Range(0,in.size()), cv::Range(0,jn.size())).copyTo(blockmask);
            cv::Range iis(ic.start-in.start, cv::min(ic.start-in.start+sz+1, in.size()));
            cv::Range jis(jc.start-jn.start, cv::min(jc.start-jn.start+sz+1, jn.size()));
            blockmask(iis, jis) = cv::Mat_<uint8_t>::zeros(cv::Size(jis.size(),iis.size()));

            cv::minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in,jn).mul(blockmask) : blockmask);
            //cv::Point cn = ijmax + cv::Point(jn.start, in.start);

            // if the block centre is also the neighbour centre, then it's a local maxima
            if (vcmax > vnmax) {
                dst.at<uint8_t>(cc.y, cc.x) = 255;
            }
        }
    }
    cv::cuda::GpuMat maxMask;
    maxMask.upload(dst, stream);
    updateParameter("Output Mask", maxMask);
    return maxMask;
}*/

// MO_REGISTER_CLASS(NonMaxSuppression, Image, Processing)
MO_REGISTER_CLASS(MinMax)
MO_REGISTER_CLASS(Threshold)
#endif
