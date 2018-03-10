#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/DetectionDescription.hpp>

namespace aq
{
    namespace nodes
    {
        class FaceDatabase: public Node
        {
        public:
            ~FaceDatabase();
            MO_DERIVE(FaceDatabase, Node)
                INPUT(DetectionDescriptionSet, detections, nullptr)
                INPUT(SyncedMemory, image, nullptr)
                PARAM(mo::ReadFile, database_path, {"./"})
                PARAM(double, min_distance, 0.5)
                OUTPUT(DetectionDescriptionSet, output, {})
            MO_END
        protected:
            virtual bool processImpl() override;
            void loadDatabase();

            aq::SyncedMemory m_facial_descriptors;
            std::shared_ptr<CategorySet> m_identities;
            std::vector<aq::SyncedMemory> m_unknown_faces;
            std::vector<cv::Mat> m_unknown_crops;
        };
    }
}
