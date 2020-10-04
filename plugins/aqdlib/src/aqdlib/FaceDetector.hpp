#ifndef AQDLIB_FACE_DETECTOR_HPP
#define AQDLIB_FACE_DETECTOR_HPP

#include <aqcore/IDetector.hpp>
#include <aqcore/detection/FaceDetector.hpp>
#define DLIB_USE_CUDA
#include <dlib/dnn.h>
#include <dlib/image_processing.h>

namespace dlib
{

    template <long num_filters, typename SUBNET>
    using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
    template <long num_filters, typename SUBNET>
    using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

    template <typename SUBNET>
    using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
    template <typename SUBNET>
    using rcon5 = relu<affine<con5<45, SUBNET>>>;

    using mmod_net_type =
        loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
} // namespace dlib

namespace aqdlib
{

    class DlibMMODDetector : public aq::nodes::FaceDetector
    {
      public:
        MO_DERIVE(DlibMMODDetector, IImageDetector)
            PARAM(mo::ReadFile, model_file, {})
        MO_END;

      protected:
        bool processImpl() override;

      private:
        std::unique_ptr<dlib::mmod_net_type> m_net;
    };

} // namespace aqdlib
#endif // AQDLIB_FACE_DETECTOR_HPP