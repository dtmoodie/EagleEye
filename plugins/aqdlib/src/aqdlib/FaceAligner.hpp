#ifndef AQDLIB_FACE_ALIGNER_HPP
#define AQDLIB_FACE_ALIGNER_HPP
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/SyncedImage.hpp>

#define DLIB_USE_CUDA
#include <dlib/image_processing/shape_predictor.h>

namespace aqdlib
{
    class FaceAligner : public aq::nodes::Node
    {
      public:
        using InputComponents_t = ct::VariadicTypedef<aq::detection::BoundingBox2d>;
        using OutputComponents_t = ct::VariadicTypedef<aq::detection::BoundingBox2d, aq::detection::LandmarkDetection>;

        MO_DERIVE(FaceAligner, aq::nodes::Node)
            INPUT(aq::SyncedImage, image)
            INPUT(aq::TDetectedObjectSet<InputComponents_t>, detections)

            PARAM(float, min_size, 20)
            PARAM(mo::ReadFile, shape_landmark_file, {"shape_predictor_5_face_landmarks.dat"});

            OUTPUT(aq::TDetectedObjectSet<OutputComponents_t>, output)
        MO_END;

      protected:
        bool processImpl() override;

      private:
        dlib::shape_predictor m_face_aligner;
        bool m_initialized = false;
    };
} // namespace aqdlib
#endif // AQDLIB_FACE_ALIGNER_HPP