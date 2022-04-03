#include "kcf.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectTrackingUtils.hpp>

namespace aqcore
{
    bool TrackerKCF::processImpl() { return false; }

    bool TrackerKCF::processImpl(mo::IAsyncStream& stream)
    {
        cv::Mat img = image->getMat(&stream);
        auto boxes = detections->getComponent<aq::detection::BoundingBox2d>();
        auto IDs = detections->getComponent<aq::detection::Id>();
        const size_t num_dets = boxes.getShape()[0];
        if (!m_tracker)
        {
            cv::TrackerKCF::Params params;
            params.detect_thresh = detect_thresh;
            params.sigma = sigma;
            params.lambda = lambda;
            params.interp_factor = interp_factor;
            params.output_sigma_factor = output_sigma_factor;
            params.pca_learning_rate = pca_learning_rate;
            params.resize = resize;
            params.split_coeff = split_coeff;
            params.wrap_kernel = wrap_kernel;
            params.compress_feature = compress_feature;
            params.max_patch_size = max_patch_size;
            params.compressed_size = compressed_size;
            params.desc_pca = desc_pca;
            params.desc_npca = desc_npca;
            m_tracker = cv::TrackerKCF::create(params);
            m_tracker->init(img, boxes[0]);
            m_previous_id = IDs[0];
            return true;
        }
        cv::Rect bb;
        const bool success = m_tracker->update(img, bb);

        auto output = *detections;

        if (success)
        {
            // now we iterate over all current detections and associate to a current detection
            float best_iou = 0.0F;
            int32_t best_iou_match = -1;
            for (size_t i = 0; i < num_dets; ++i)
            {
                const float iou = aq::iou(cv::Rect2f(bb), boxes[i]);
                if (iou > iou_threshold)
                {
                    if (iou > best_iou)
                    {
                        best_iou = iou;
                        best_iou_match = i;
                    }
                }
            }
            if (best_iou_match != -1)
            {
                // update the database with the new ID,
                auto current_id = IDs[best_iou_match];
                // check the current ID
                if (current_id != m_previous_id)
                {
                    // Need to update the current ID and all other IDs in this detection set
                    auto IDs = output.getComponentMutable<aq::detection::Id>();
                    IDs[best_iou_match] = m_previous_id;

                    // Now make the ID unique
                }
            }
            else
            {
                // We have not found a match, so add this detection to the database
                // TODO propogate components for this entity from the detection
            }
        }

        this->output.publish(output, mo::tags::param = &image_param);

        return true;
    }

    bool TrackerKCF::processImpl(mo::IDeviceStream& stream)
    {
        return processImpl(static_cast<mo::IAsyncStream&>(stream));
    }
} // namespace aqcore

using namespace aqcore;
MO_REGISTER_CLASS(TrackerKCF)
