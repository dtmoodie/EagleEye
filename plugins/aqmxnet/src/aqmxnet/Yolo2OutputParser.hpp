#pragma once
#include "MXNetOutputParser.hpp"
#include "mxnet-cpp/ndarray.h"
#include <MetaObject/params/ParamMacros.hpp>

namespace aq
{
    namespace mxnet
    {

        class Yolo2OutputParser : public MXNetOutputParser
        {
          public:
            MO_DERIVE(Yolo2OutputParser, MXNetOutputParser)
                PARAM(int, num_class, 1)
                PARAM(int, num_anchor, 5)
                PARAM(std::vector<float>, detection_threshold, {0.1f})
                PARAM(float, iou_threshold, 0.2)
                OUTPUT(DetectedObjectSet, detections, {})
            MO_END

            static int parserPriority(const ::mxnet::cpp::Symbol& sym);

            virtual void setupParser(const ::mxnet::cpp::Symbol& sym,
                                     const CategorySet::ConstPtr& cats,
                                     const std::vector<::mxnet::cpp::NDArray>& outputs) override;
            virtual void preBatch(unsigned int total_batch_size) override;
            virtual void postMiniBatch(const std::vector<cv::Rect>& batch_bb, const DetectedObjectSet& dets) override;
            virtual void postBatch(mo::IParam& param) override;

          private:
            aq::CategorySet::ConstPtr m_cats;
            ::mxnet::cpp::NDArray m_gpu_output;
            ::mxnet::cpp::NDArray m_cpu_buffer;
        };
    } // namespace mxnet
} // namespace aq
