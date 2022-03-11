#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/DetectionPatch.hpp>
#include <Aquila/types/SyncedImage.hpp>
#include <Aquila/types/TSyncedImage.hpp>
#include <Aquila/types/TSyncedMemory.hpp>

#include <MetaObject/thread/Thread.hpp>
#include <ct/reflect_macros.hpp>

#include <MetaObject/runtime_reflection.hpp>

#include <boost/circular_buffer.hpp>

namespace aqdlib
{

    class FaceDatabase : public aq::nodes::Node
    {
      public:
        enum DistanceMeasurement
        {
            Euclidean = 0,
            CosineSimilarity = 1
        };

        using InputComponents_t =
            ct::VariadicTypedef<aq::detection::Descriptor, aq::detection::AlignedPatch, aq::detection::BoundingBox2d>;

        using OutputComponents_t = ct::VariadicTypedef<aq::detection::Classifications,
                                                       aq::detection::Descriptor,
                                                       aq::detection::AlignedPatch,
                                                       aq::detection::BoundingBox2d,
                                                       aq::detection::Confidence,
                                                       aq::detection::Id>;

        using Input_t = aq::TDetectedObjectSet<InputComponents_t>;
        using Output_t = aq::TDetectedObjectSet<OutputComponents_t>;

        ~FaceDatabase();
        MO_DERIVE(FaceDatabase, Node)
            INPUT(Input_t, detections)
            INPUT(aq::SyncedImage, image)

            PARAM(mo::ReadDirectory, database_path, {"./"})
            PARAM(mo::AppendDirectory, unknown_detections, {"./", "unknown", ".jpg"})
            PARAM(mo::AppendDirectory, known_detections, {"./", "known", ".jpg"})
            PARAM(mo::AppendDirectory, recent_detections, {"./", "recent", ".jpg"})

            PARAM(double, min_distance, 0.5)
            ENUM_PARAM(distance_measurement, Euclidean, CosineSimilarity)
            PARAM(uint32_t, patch_buffer_size, 100)

            MO_SLOT(void, saveUnknownFaces)
            MO_SLOT(void, saveKnownFaces)
            MO_SLOT(void, saveRecentFaces)
            MO_SLOT(void, loadDatabase)

            MO_SIGNAL(void, detectedUnknownFace, aq::SyncedImage, aq::detection::Descriptor, int)
            MO_SIGNAL(void, detectedKnownFace, aq::SyncedImage, aq::detection::Descriptor)

            OUTPUT(Input_t, output, {})
        MO_END;

        struct IdentityDatabase
        {
            IdentityDatabase();
            IdentityDatabase(const std::vector<aq::TSyncedMemory<float>>& unkown, mo::IAsyncStreamPtr_t);

            std::vector<std::string> identities;
            aq::TSyncedImage<aq::GRAY<float>> descriptors;
            std::vector<int> membership;

            void load(mo::ILoadVisitor&, const std::string&);
            void save(mo::ISaveVisitor&, const std::string&) const;
        };

        struct ClassifiedPatch
        {
            aq::SyncedImage patch;
            aq::TSyncedMemory<float> embeddings;
            std::string classification;
            void save(const std::string file_path, const int index, mo::IAsyncStream& stream) const;
        };

      protected:
        bool processImpl() override;

        bool matchKnownFaces(const mt::Tensor<const float, 1>& descriptor,
                             aq::detection::Classifications& cls,
                             aq::detection::Id::DType& id,
                             const double mag0,
                             const aq::SyncedImage& patch,
                             mo::IAsyncStream& stream);

        bool matchUnknownFaces(const mt::Tensor<const float, 1>& descriptor,
                               aq::detection::Classifications& cls,
                               aq::detection::Id::DType& id,
                               const double mag0,
                               const aq::SyncedImage& patch,
                               mo::IAsyncStream& stream);

        void onNewUnknownFace(const mt::Tensor<const float, 1>& det_desc,
                              aq::detection::Classifications& cls,
                              aq::detection::Id::DType& id,
                              const aq::SyncedImage& patch);

        void saveUnknownFace(uint32_t index);

        void nodeInit(bool) override;

        std::shared_ptr<aq::CategorySet> m_identities;
        std::vector<aq::TSyncedMemory<float>> m_unknown_face_descriptors;
        std::vector<aq::SyncedImage> m_unknown_crops;
        std::vector<int> m_unknown_det_count;
        std::map<std::string, boost::circular_buffer<cv::Mat>> m_known_face_patches;
        IdentityDatabase m_known_faces;
        int m_unknown_write_count = 0;
        boost::circular_buffer<ClassifiedPatch> m_recent_patches;
        mo::Thread m_worker_thread;
        mo::IAsyncStreamPtr_t m_worker_stream;
    };

} // namespace aqdlib

namespace ct
{
    REFLECT_BEGIN(aqdlib::FaceDatabase::IdentityDatabase)
    PUBLIC_ACCESS(identities)
    PUBLIC_ACCESS(descriptors)
    PUBLIC_ACCESS(membership)
    REFLECT_END;

    REFLECT_BEGIN(aqdlib::FaceDatabase::ClassifiedPatch)
    PUBLIC_ACCESS(patch)
    PUBLIC_ACCESS(embeddings)
    PUBLIC_ACCESS(classification)
    REFLECT_END;
} // namespace ct
