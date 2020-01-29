#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "Aquila/nodes/NodeInfo.hpp"
#include "flip.hpp"
#include "opencv2/imgproc.hpp"
#include <Aquila/types/ObjectDetection.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

cv::Rect constrainRoi(cv::Rect roi, cv::Size size)
{
    return roi & cv::Rect(cv::Point(0, 0), size);
}

using namespace aq::nodes;
bool Flip::processImpl()
{
    // TODO figure out if we are on tk1
    const bool tk1 = true;
    cv::Rect2f bb_ = roi;
    const cv::Size size = input->getSize();
    boundingBoxToPixels(bb_, size);
    auto bb = constrainRoi(bb_, size);
    if (tk1 || _ctx->device_id == -1)
    {
        cv::Mat source = input->getMatNoSync();
        source = source(bb);
        cv::Mat flipped;
        if (axis.getValue() == X)
        {
            flipped.create(source.size(), source.type());
            const size_t src_stride = source.step;
            const size_t dst_stride = flipped.step;
            const size_t row_width = source.cols * source.elemSize();
            uchar* src = source.data;
            uchar* dst = flipped.data;
            dst += dst_stride * source.rows;
            for (int i = 0; i < source.rows; ++i)
            {
                memcpy(dst, src, row_width);
                src += src_stride;
                dst -= dst_stride;
            }
        }
        else
        {
            cv::flip(source, flipped, axis.getValue());
        }

        output_param.updateData(flipped, mo::tag::_param = input_param);
        return true;
    }
    else
    {
        cv::cuda::GpuMat output;

        if (axis.getValue() == X)
        {
            if (input->getSyncState() == input->HOST_UPDATED)
            {
                auto source = input->getMat(stream());
                output.create(source.size(), source.type());
                const size_t src_stride = source.step;
                const size_t dst_stride = output.step;
                uchar* src = source.data;
                uchar* dst = output.data;
                dst += dst_stride * (source.rows - 1);
                cudaStream_t strm = _ctx->getCudaStream();
                for (int i = 0; i < source.rows; ++i)
                {
                    auto err = cudaMemcpyAsync(dst, src, src_stride, cudaMemcpyHostToDevice, strm);
                    src += src_stride;
                    dst -= dst_stride;
                }
            }
            else
            {
                auto source = input->getGpuMat(stream());
                output.create(source.size(), source.type());
                const size_t src_stride = source.step;
                const size_t dst_stride = output.step;
                uchar* src = source.data;
                uchar* dst = output.data;
                dst += dst_stride * source.rows;
                cudaStream_t strm = _ctx->getCudaStream();
                for (int i = 0; i < source.rows; ++i)
                {
                    cudaMemcpyAsync(dst, src, src_stride, cudaMemcpyDeviceToDevice, strm);
                    src += src_stride;
                    dst -= dst_stride;
                }
            }
        }
        else
        {
            auto source = input->getGpuMat(stream());
            cv::cuda::flip(source, output, axis.getValue(), stream());
        }

        output_param.updateData(output, mo::tag::_param = input_param);
        return true;
    }
    return false;
}

MO_REGISTER_CLASS(Flip)

bool Rotate::processImpl()
{
    cv::cuda::GpuMat rotated;
    auto size = input->getSize();
    cv::Mat rotation = cv::getRotationMatrix2D({size.width / 2.0f, size.height / 2.0f}, angle_degrees, 1.0);
    cv::cuda::warpAffine(input->getGpuMat(stream()),
                         rotated,
                         rotation,
                         size,
                         cv::INTER_CUBIC,
                         cv::BORDER_REFLECT,
                         cv::Scalar(),
                         stream());
    output_param.updateData(rotated, input_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(Rotate)
#endif
