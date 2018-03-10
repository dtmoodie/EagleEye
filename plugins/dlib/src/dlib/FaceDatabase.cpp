#include "FaceDatabase.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
namespace aq
{
    namespace nodes
    {
        FaceDatabase::~FaceDatabase()
        {
            std::ofstream ofs;
            ofs.open(database_path.string() + "/unknown.db");
            cereal::JSONOutputArchive ar(ofs);
            ar(cereal::make_nvp("descriptors", m_unknown_faces));
            for(size_t i = 0; i < m_unknown_crops.size(); ++i)
            {
                cv::imwrite(database_path.string() + std::to_string(i) + ".jpg", m_unknown_crops[i]);
            }
        }

        bool FaceDatabase::processImpl()
        {
            if(m_facial_descriptors.empty())
            {
                loadDatabase();
            }
            output.clear();
            output.setCatSet(m_identities);
            cv::Mat host_img;
            if(_ctx->device_id == -1)
            {
                host_img = image->getMatNoSync();
            }
            for(const auto& det : *detections)
            {
                DetectionDescription out_det(det);
                if(_ctx->device_id == -1)
                {
                    cv::Mat det_desc = det.descriptor.getMatNoSync();
                    int match_index = -1;
                    double best_match = 1000;
                    if(!m_facial_descriptors.empty())
                    {
                        cv::Mat db_desc = m_facial_descriptors.getMatNoSync();
                        for(int i = 0; i < db_desc.rows; ++i)
                        {
                            auto dist = cv::norm(db_desc.row(i) - det_desc);
                            if(dist < best_match)
                            {
                                best_match = dist;
                                match_index = static_cast<int>(i);
                            }
                        }
                        if(best_match < min_distance)
                        {
                            out_det.classifications.resize(1);
                            out_det.classifications[0] = (*m_identities)[static_cast<size_t>(match_index+1)]();
                        }else
                        {
                            match_index = -1;
                            out_det.classifications.resize(1);
                            out_det.classifications[0] = (*m_identities)[static_cast<size_t>(0)]();
                        }
                    }
                    if(match_index == -1)
                    {
                        // match to the unknown faces
                        for(size_t i = 0; i < m_unknown_faces.size(); ++i)
                        {
                            cv::Mat desc = m_unknown_faces[i].getMatNoSync();
                            auto dist = cv::norm(desc - det_desc);
                            if(dist < best_match)
                            {
                                best_match = dist;
                                match_index = static_cast<int>(i);
                            }
                        }
                        if(match_index == -1 || best_match > min_distance)
                        {
                            m_unknown_faces.push_back(det_desc);
                            m_unknown_crops.push_back(host_img(det.bounding_box));
                        }
                    }
                }
                output.emplace_back(std::move(out_det));
            }
            output_param.emitUpdate(detections_param);
            return true;
        }

        void FaceDatabase::loadDatabase()
        {
            if(boost::filesystem::exists(database_path.string() + "/identities.db"))
            {
                std::ifstream ifs;
                ifs.open(database_path.string() + "/identities.db");
                cereal::JSONInputArchive ar(ifs);
                std::vector<SyncedMemory> descriptors;
                std::vector<std::string> identities;
                ar(CEREAL_NVP(descriptors));
                ar(CEREAL_NVP(identities));
                MO_ASSERT(descriptors.size());
                int width = descriptors[0].getSize().width;
                cv::Mat host_descriptors;
                host_descriptors.create(static_cast<int>(descriptors.size()), width, CV_32F);
                for(size_t i = 0;i < descriptors.size(); ++i)
                {
                    cv::Mat desc = descriptors[i].getMatNoSync();
                    desc.copyTo(host_descriptors.row(static_cast<int>(i)));
                }
                identities.insert(identities.begin(), "unknown");
                m_identities = std::make_shared<CategorySet>(identities);
                m_facial_descriptors = SyncedMemory(host_descriptors);
            }
        }

    }
}

using namespace aq::nodes;
MO_REGISTER_CLASS(FaceDatabase)
