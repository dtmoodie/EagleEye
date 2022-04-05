#ifndef AQCORE_KCF_HPP
#define AQCORE_KCF_HPP
#include <ct/types/opencv.hpp>

#include <Aquila/nodes/Node.hpp>

#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <MetaObject/params/ParamMacros.hpp>

#include <opencv2/tracking/tracking.hpp>

namespace aqcore
{
    class TrackerKCF : public aq::nodes::Node
    {
      public:
        using Components_t = ct::VariadicTypedef<aq::detection::BoundingBox2d, aq::detection::Id>;
        using Input_t = aq::TDetectedObjectSet<Components_t>;
        using Output_t = Input_t;

        MO_DERIVE(TrackerKCF, aq::nodes::Node)
            INPUT(aq::SyncedImage, image)
            INPUT(Input_t, detections)

            OUTPUT(Output_t, output)

            PARAM(float, detect_thresh, 0.5F)               //!<  detection confidence threshold
            PARAM(float, sigma, 0.2F)                       //!<  gaussian kernel bandwidth
            PARAM(float, lambda, 0.0001F)                   //!<  regularization
            PARAM(float, interp_factor, 0.075F)             //!<  linear interpolation factor for adaptation
            PARAM(float, output_sigma_factor, 1.0F / 16.0F) //!<  spatial bandwidth (proportional to target)
            PARAM(float, pca_learning_rate, 0.15F)          //!<  compression learning rate
            PARAM(bool, resize, true)           //!<  activate the resize feature to improve the processing speed
            PARAM(bool, split_coeff, true)      //!<  split the training coefficients into two matrices
            PARAM(bool, wrap_kernel, false)     //!<  wrap around the kernel values
            PARAM(bool, compress_feature, true) //!<  activate the pca method to compress the features
            PARAM(int, max_patch_size, 80 * 80) //!<  threshold for the ROI size
            PARAM(int, compressed_size, 2)      //!<  feature size after compression
            PARAM(int, desc_pca, cv::TrackerKCF::MODE::CN)    //!<  compressed descriptors of TrackerKCF::MODE
            PARAM(int, desc_npca, cv::TrackerKCF::MODE::GRAY) //!<  non-compressed descriptors of TrackerKCF::MODE

            PARAM(float, iou_threshold, 0.6F)
        MO_END;

        bool processImpl() override;
        bool processImpl(mo::IAsyncStream& stream) override;
        bool processImpl(mo::IDeviceStream& stream) override;

      private:
        cv::Ptr<cv::TrackerKCF> m_tracker;
        boost::optional<mo::Header> m_previous_header;
        // Should save more than just the previous ID, we should save all components
        aq::detection::Id m_previous_id;
        aq::DetectedObjectSet m_previous_detection;
    };
} // namespace aqcore

#endif // AQCORE_KCF_HPP
