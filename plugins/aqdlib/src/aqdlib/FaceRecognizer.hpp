#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/SyncedImage.hpp>
#define DLIB_USE_CUDA
#include <dlib/dnn.h>

namespace dlib
{
    template <template <int, template <typename> class, int, typename> class block,
              int N,
              template <typename> class BN,
              typename SUBNET>
    using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

    template <template <int, template <typename> class, int, typename> class block,
              int N,
              template <typename> class BN,
              typename SUBNET>
    using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

    template <int N, typename SUBNET>
    using ares = relu<residual<block, N, affine, SUBNET>>;
    template <int N, typename SUBNET>
    using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

    template <typename SUBNET>
    using alevel0 = ares_down<256, SUBNET>;
    template <typename SUBNET>
    using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
    template <typename SUBNET>
    using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
    template <typename SUBNET>
    using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
    template <typename SUBNET>
    using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
    using anet_type = loss_metric<fc_no_bias<
        128,
        avg_pool_everything<alevel0<alevel1<alevel2<
            alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;
} // namespace dlib

namespace aqdlib
{

    class FaceRecognizer : public aq::nodes::Node
    {
      public:
        using InputComponents_t = ct::VariadicTypedef<aq::detection::BoundingBox2d, aq::detection::LandmarkDetection>;
        using OutputComponents_t = ct::
            VariadicTypedef<aq::detection::BoundingBox2d, aq::detection::LandmarkDetection, aq::detection::Descriptor>;

        MO_DERIVE(FaceRecognizer, aq::nodes::Node)
            INPUT(aq::SyncedImage, image)
            INPUT(aq::TDetectedObjectSet<InputComponents_t>, detections)

            PARAM(mo::ReadFile, face_recognizer_weight_file, {"dlib_face_recognition_resnet_model_v1.dat"})

            OUTPUT(aq::TDetectedObjectSet<OutputComponents_t>, output)
        MO_END;

      protected:
        bool processImpl() override;
        dlib::anet_type m_net;
        bool m_initialized = false;
    };

} // namespace aqdlib
