#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/DetectionDescription.hpp>

namespace aq
{
    namespace nodes
    {
        class FaceDatabase : public Node
        {
          public:
            ~FaceDatabase();
            MO_DERIVE(FaceDatabase, Node)
                INPUT(DetectionDescriptionSet, detections, nullptr)
                INPUT(SyncedMemory, image, nullptr)

                PARAM(mo::ReadDirectory, database_path, {"./"})
                PARAM(double, min_distance, 0.5)

                MO_SLOT(void, saveUnknownFaces)
                MO_SLOT(void, saveKnownFaces)

                MO_SIGNAL(void, detectedUnknownFace, aq::SyncedMemory, DetectionDescription, int)
                MO_SIGNAL(void, detectedKnownFace, aq::SyncedMemory, DetectionDescription)

                OUTPUT(DetectionDescriptionSet, output, {})
            MO_END
          protected:
            virtual bool processImpl() override;
            void loadDatabase();

            struct IdentityDatabase
            {
                IdentityDatabase();
                IdentityDatabase(const std::vector<SyncedMemory> unkown);

                template <class AR>
                void save(AR&) const;
                template <class AR>
                void load(AR&);
                void save(cereal::JSONOutputArchive& ar) const;
                void load(cereal::JSONInputArchive& ar);

                std::vector<std::string> identities;
                aq::SyncedMemory descriptors;
            };

            std::shared_ptr<CategorySet> m_identities;
            std::vector<SyncedMemory> m_unknown_faces;
            std::vector<cv::Mat> m_unknown_crops;
            std::vector<int> m_unknown_det_count;
            IdentityDatabase m_known_faces;
            int m_unknown_write_count = 0;
        };
    }
}
