#include "Yolo2OutputParser.hpp"
#include "MXNetOutputParserInfo.hpp"
#include "mxnet-cpp/ndarray.hpp"
#include "mxnet-cpp/symbol.h"
#include "mxnet/c_api.h"
#include <Aquila/types/ObjectTracking.hpp>
#include <opencv2/core.hpp>

namespace aq
{

    namespace mxnet
    {

        int Yolo2OutputParser::parserPriority(const ::mxnet::cpp::Symbol& sym)
        {
            const char* name = nullptr;
            int success = 0;
            if (MXSymbolGetName(sym.GetHandle(), &name, &success) == 0)
            {
                if (success)
                {
                    std::string name_str = name;
                    if (name_str.find("yolo_output") != std::string::npos)
                    {
                        return 10;
                    }
                }
            }
            return 0;
        }

        void Yolo2OutputParser::setupParser(const ::mxnet::cpp::Symbol& sym,
                                            const CategorySet::ConstPtr& cats,
                                            const std::vector<::mxnet::cpp::NDArray>& outputs)
        {
            readAttr(sym, "num_class", num_class);
            readAttr(sym, "num_anchor", num_anchor);
            m_cats = cats;
            m_gpu_output = outputs[0];
            auto ctx = ::mxnet::cpp::Context::cpu(0);
            m_cpu_buffer = m_gpu_output.Copy(ctx);
        }

        void Yolo2OutputParser::preBatch(unsigned int /*total_batch_size*/)
        {
            detections.clear();
            detections.setCatSet(m_cats);
        }

        cv::Mat_<float> wrapOutput(const ::mxnet::cpp::NDArray& cpu_data)
        {
            auto shape = cpu_data.GetShape();
            CHECK_EQ(shape.size(), 3);
            CHECK_EQ(shape[0], 1);
            int num_dets = static_cast<int>(shape[1]);
            int num_fields = static_cast<int>(shape[2]);
            float* ptr = const_cast<float*>(cpu_data.GetData());
            return cv::Mat_<float>(num_dets, num_fields, ptr);
        }

        void Yolo2OutputParser::postMiniBatch(const std::vector<cv::Rect>& batch_bb, const DetectedObjectSet& /*dets*/)
        {
            m_gpu_output.CopyTo(&m_cpu_buffer);
            m_cpu_buffer.WaitToRead();
            cv::Mat_<float> mat = wrapOutput(m_cpu_buffer.Slice(0, 1));
            const int num_dets = mat.rows;
            int det_count = 0;
            for (int i = 0; i < num_dets; ++i)
            {
                float* ptr = mat.ptr<float>(i);
                int cat = static_cast<int>(ptr[0]);
                if (cat >= 0)
                {
                    float conf = ptr[1];
                    bool good = false;
                    if (detection_threshold.size() == 1)
                    {
                        if (conf > detection_threshold[0])
                        {
                            good = true;
                        }
                    }
                    else
                    {
                        if (cat >= 0 && cat < detection_threshold.size())
                        {
                            if (conf > detection_threshold[cat])
                            {
                                good = true;
                            }
                        }
                    }
                    if (good)
                    {
                        float x0 = ptr[2];
                        float y0 = ptr[3];
                        float x1 = ptr[4];
                        float y1 = ptr[5];
                        x0 *= batch_bb[0].width;
                        x1 *= batch_bb[0].width;
                        y0 *= batch_bb[0].height;
                        y1 *= batch_bb[0].height;
                        cv::Rect2f bb(cv::Point2f(x0, y0), cv::Point2f(x1, y1));
                        bool insert = true;
                        for (auto& det : detections)
                        {
                            float iou = aq::iou(det.bounding_box, bb);
                            if (iou > iou_threshold)
                            {
                                if (conf > det.confidence)
                                {
                                    det.bounding_box = bb;
                                    det.classifications = (*m_cats)[cat](1.0f);
                                    insert = false;
                                    break;
                                }
                            }
                        }
                        if (insert)
                        {
                            aq::DetectedObject det(bb, (*m_cats)[cat](1.0f), det_count, conf);
                            detections.emplace_back(std::move(det));
                            ++det_count;
                        }
                    }
                }
            }
        }

        void Yolo2OutputParser::postBatch(mo::IParam& param) { detections_param.emitUpdate(param); }

    } // namespace mxnet

} // namespace aq
using namespace aq::mxnet;

MO_REGISTER_CLASS(Yolo2OutputParser)
