#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/DetectionDescription.hpp>
#include <boost/circular_buffer.hpp>

namespace aq
{
namespace nodes
{
class FaceDatabase : public Node
{
  public:
    enum DistanceMeasurement
    {
        Euclidean = 0,
        CosineSimilarity = 1
    };

    ~FaceDatabase();
    MO_DERIVE(FaceDatabase, Node)
        INPUT(DetectionDescriptionPatchSet, detections, nullptr)
        INPUT(SyncedMemory, image, nullptr)

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

        MO_SIGNAL(void, detectedUnknownFace, aq::SyncedMemory, DetectionDescription, int)
        MO_SIGNAL(void, detectedKnownFace, aq::SyncedMemory, DetectionDescription)

        OUTPUT(DetectionDescriptionPatchSet, output, {})
    MO_END
  protected:
    virtual bool processImpl() override;

    struct IdentityDatabase
    {
        IdentityDatabase();
        IdentityDatabase(const std::vector<SyncedMemory>& unkown);

        template <class AR>
        void save(AR&) const;
        template <class AR>
        void load(AR&);
        void save(cereal::JSONOutputArchive& ar) const;
        void load(cereal::JSONInputArchive& ar);

        std::vector<std::string> identities;
        aq::SyncedMemory descriptors;
        std::vector<int> membership;
    };

    std::shared_ptr<CategorySet> m_identities;
    std::vector<SyncedMemory> m_unknown_faces;
    std::vector<cv::Mat> m_unknown_crops;
    std::vector<int> m_unknown_det_count;
    std::map<std::string, boost::circular_buffer<cv::Mat>> m_known_face_patches;
    IdentityDatabase m_known_faces;
    int m_unknown_write_count = 0;

    struct ClassifiedPatch
    {
        aq::SyncedMemory patch;
        aq::SyncedMemory embeddings;
        std::string classification;
        void save(const std::string file_path, const int index, cv::cuda::Stream& stream);
    };
    boost::circular_buffer<ClassifiedPatch> m_recent_patches;
};
}
}
