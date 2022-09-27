
#include <MetaObject/core/metaobject_config.hpp>

#include "NonMaxSuppression.h"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>

using namespace aq;
using namespace aq::nodes;

namespace aqcore
{
    bool MinMax::processImpl(aq::CVStream& stream)
    {
        double min, max;
        stream.synchronize();
        cv::cuda::minMax(input->getGpuMat(&stream), &min, &max);
        min_value.publish(min, mo::tags::param = &input_param, mo::tags::stream = &stream);
        max_value.publish(max, mo::tags::param = &input_param, mo::tags::stream = &stream);
        return true;
    }

    bool MinMax::processImpl(mo::IAsyncStream& stream)
    {
        if (stream.isDeviceStream())
        {
            if (this->processImpl(*stream.getDeviceStream()))
            {
                return true;
            }
        }
        cv::Mat in = input->getMat(&stream);
        double min, max;
        int32_t min_idx, max_idx;
        cv::minMaxIdx(in, &min, &max, &min_idx, &max_idx);
        min_value.publish(min, mo::tags::param = &input_param, mo::tags::stream = &stream);
        max_value.publish(max, mo::tags::param = &input_param, mo::tags::stream = &stream);
        return true;
    }

    bool Threshold::processImpl(aq::CVStream& stream)
    {
        if (input_max)
        {
            max_param.setValue(*input_max * input_percent);
        }
        if (input_min)
        {
            min_param.setValue(*input_min * input_percent);
        }
        cv::cuda::GpuMat upper_mask, lower_mask;
        bool sync = false;
        cv::cuda::GpuMat in = input->getGpuMat(&stream, &sync);
        cv::cuda::Stream& cvstream = stream.getCVStream();
        if (two_sided)
        {
            if (source_value)
            {
                cv::cuda::threshold(in, upper_mask, max, replace_value, inverse ? 3 : 4, cvstream);
            }
            else
            {
                cv::cuda::threshold(in, upper_mask, max, replace_value, inverse ? 1 : 0, cvstream);
            }
        }
        if (truncate)
        {
            cv::cuda::threshold(in, lower_mask, min, replace_value, 2, cvstream);
        }
        else
        {
            if (source_value)
            {
                cv::cuda::threshold(in, lower_mask, min, 0.0, inverse ? 4 : 3, cvstream);
            }
            else
            {
                cv::cuda::threshold(in, lower_mask, min, replace_value, inverse ? 1 : 0, cvstream);
            }
        }
        cv::cuda::GpuMat mask;
        if (upper_mask.empty())
        {
            mask = lower_mask;
        }
        else
        {
            cv::cuda::bitwise_and(upper_mask, lower_mask, mask, cv::noArray(), cvstream);
        }
        mask.convertTo(mask, in.type(), cvstream);
        this->output.publish(mask, mo::tags::param = &input_param, mo::tags::stream = &stream);
        return true;
    }

    bool Threshold::processImpl(mo::IAsyncStream& stream)
    {
        if (stream.isDeviceStream())
        {
            return this->processImpl(*stream.getDeviceStream());
        }
        // TODO impelment
        return false;
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
} // namespace aqcore
// MO_REGISTER_CLASS(NonMaxSuppression, Image, Processing)
using namespace aqcore;
MO_REGISTER_CLASS(MinMax)
MO_REGISTER_CLASS(Threshold)
