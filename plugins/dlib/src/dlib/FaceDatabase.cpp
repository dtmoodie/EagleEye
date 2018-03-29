#include "FaceDatabase.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

namespace aq
{
    namespace nodes
    {
        FaceDatabase::IdentityDatabase::IdentityDatabase() {}

        FaceDatabase::IdentityDatabase::IdentityDatabase(const std::vector<SyncedMemory> unknown)
        {
            cv::Mat desc(unknown.size(), unknown[0].getSize().width, CV_32F);
            for (size_t i = 0; i < unknown.size(); ++i)
            {
                unknown[i].getMatNoSync().copyTo(desc.row(i));
                identities.push_back("unknown" + std::to_string(i));
            }
            descriptors = SyncedMemory(desc);
        }

        template <class AR>
        void FaceDatabase::IdentityDatabase::save(AR& ar) const
        {
            ar(CEREAL_NVP(identities));
            cv::Mat descriptors = this->descriptors.getMatNoSync();
            ar(cereal::make_nvp("descriptor_width", descriptors.cols));
            ar(cereal::make_nvp(
                "descriptors",
                cereal::binary_data(reinterpret_cast<float*>(descriptors.data), descriptors.rows * descriptors.cols)));
        }

        template <class AR>
        void FaceDatabase::IdentityDatabase::load(AR& ar)
        {
            ar(CEREAL_NVP(identities));
            int cols;
            ar(cereal::make_nvp("descriptor_width", cols));
            cv::Mat desc(identities.size(), cols, CV_32F);
            ar(cereal::make_nvp("descriptors",
                                cereal::binary_data(reinterpret_cast<float*>(desc.data), desc.rows * desc.cols)));
            descriptors = SyncedMemory(desc);
        }

        void FaceDatabase::IdentityDatabase::save(cereal::JSONOutputArchive& ar) const
        {
            if (!descriptors.empty())
            {
                cv::Mat descriptors = this->descriptors.getMatNoSync();
                ar(cereal::make_nvp("descriptor_width", descriptors.cols));
                for (size_t i = 0; i < identities.size(); ++i)
                {
                    ar.saveBinaryValue(
                        descriptors.ptr<float>(i), sizeof(float) * descriptors.cols, identities[i].c_str());
                }
            }
        }

        void FaceDatabase::IdentityDatabase::load(cereal::JSONInputArchive& ar)
        {
            int cols;
            ar(cereal::make_nvp("descriptor_width", cols));
            std::vector<cv::Mat> rows;
            identities.clear();
            while (true)
            {
                const auto name = ar.getNodeName();
                if (!name)
                    break;
                cv::Mat row(1, cols, CV_32F);
                ar.loadBinaryValue(row.data, sizeof(float) * cols);
                identities.push_back(name);
                rows.push_back(row);
            }
            cv::Mat desc(identities.size(), cols, CV_32F);
            for (size_t i = 0; i < identities.size(); ++i)
            {
                rows[i].copyTo(desc.row(i));
            }
            descriptors = SyncedMemory(desc);
        }

        FaceDatabase::~FaceDatabase()
        {
            // saveUnknownFaces();
        }

        void FaceDatabase::saveUnknownFaces()
        {
            std::ofstream ofs;
            ofs.open(database_path.string() + "/unknown.db");
            if (m_unknown_crops.size())
            {
                cereal::JSONOutputArchive ar(ofs);
                IdentityDatabase unknown(m_unknown_faces);
                ar(cereal::make_nvp("unknown", unknown));
                for (size_t i = 0; i < m_unknown_crops.size(); ++i)
                {
                    cv::imwrite(database_path.string() + std::to_string(i) + ".jpg", m_unknown_crops[i]);
                }
            }
        }

        void FaceDatabase::saveKnownFaces()
        {
            mo::Mutex_t::scoped_lock lock(getMutex());
            std::ofstream ofs;
            ofs.open(database_path.string() + "/identities.db");
            cereal::JSONOutputArchive ar(ofs);
            ar(cereal::make_nvp("face_db", m_known_faces));
        }

        bool FaceDatabase::processImpl()
        {
            if (m_known_faces.descriptors.empty())
            {
                loadDatabase();
            }
            output.clear();
            output.setCatSet(m_identities);
            cv::Mat host_img;
            if (_ctx->device_id == -1)
            {
                host_img = image->getMatNoSync();
            }
            else
            {
                bool sync = false;
                host_img = image->getMat(stream(), 0, &sync);
                if (sync)
                    stream().waitForCompletion();
            }
            for (const auto& det : *detections)
            {
                DetectionDescription out_det(det);
                cv::Mat det_desc;
                if (_ctx->device_id == -1)
                {
                    det_desc = det.descriptor.getMatNoSync();
                }
                else
                {
                    bool sync = false;
                    det_desc = det.descriptor.getMat(stream(), 0, &sync);
                    if (sync)
                        stream().waitForCompletion();
                }

                int match_index = -1;
                double best_match = 1000;
                if (!m_known_faces.descriptors.empty())
                {
                    cv::Mat db_desc = m_known_faces.descriptors.getMatNoSync();
                    for (int i = 0; i < db_desc.rows; ++i)
                    {
                        auto dist = cv::norm(db_desc.row(i) - det_desc);
                        if (dist < best_match)
                        {
                            best_match = dist;
                            match_index = static_cast<int>(i);
                        }
                    }
                    if (best_match < min_distance)
                    {
                        out_det.classifications.resize(1);
                        out_det.classifications[0] = (*m_identities)[static_cast<size_t>(match_index + 1)]();
                        sig_detectedKnownFace(*image, out_det);
                    }
                    else
                    {
                        match_index = -1;
                        out_det.classifications.resize(1);
                        out_det.classifications[0] = (*m_identities)[static_cast<size_t>(0)]();
                    }
                }
                if (match_index == -1)
                {
                    // match to the unknown faces
                    for (size_t i = 0; i < m_unknown_faces.size(); ++i)
                    {
                        cv::Mat desc = m_unknown_faces[i].getMatNoSync();
                        auto dist = cv::norm(desc - det_desc);
                        if (dist < best_match)
                        {
                            best_match = dist;
                            match_index = static_cast<int>(i);
                        }
                    }
                    if (match_index == -1 || best_match > min_distance)
                    {
                        m_unknown_faces.push_back(det_desc);
                        m_unknown_crops.push_back(host_img(det.bounding_box).clone());
                        m_unknown_det_count.push_back(1);
                    }
                    else
                    {
                        ++m_unknown_det_count[match_index];
                        if (m_unknown_det_count[match_index] == 10)
                        {
                            auto desc = m_unknown_faces[match_index].getMatNoSync();
                            std::string base64 = cereal::base64::encode(desc.data, desc.cols * sizeof(float));
                            std::string stem =
                                database_path.string() + "/unknown_" + std::to_string(m_unknown_write_count);
                            m_unknown_write_count++;
                            std::ofstream ofs(stem + ".bin");
                            ofs << base64;
                            cv::imwrite(stem + ".jpg", m_unknown_crops[match_index]);
                        }
                        sig_detectedUnknownFace(*image, out_det, m_unknown_det_count[match_index]);
                    }
                }
                output.emplace_back(std::move(out_det));
            }
            output_param.emitUpdate(detections_param);
            return true;
        }

        void FaceDatabase::loadDatabase()
        {
            bool loaded = false;
            if (boost::filesystem::exists(database_path.string() + "/identities.bin"))
            {
                std::ifstream ifs;
                ifs.open(database_path.string() + "/identities.bin");
                cereal::BinaryInputArchive ar(ifs);
                ar(cereal::make_nvp("face_db", m_known_faces));
                loaded = true;
            }
            if (boost::filesystem::exists(database_path.string() + "/identities.db"))
            {
                std::ifstream ifs;
                ifs.open(database_path.string() + "/identities.db");
                cereal::JSONInputArchive ar(ifs);
                ar(cereal::make_nvp("face_db", m_known_faces));
                loaded = true;
            }
            if (loaded)
            {
                auto tmp = m_known_faces.identities;
                tmp.insert(tmp.begin(), "unknown");
                m_identities = std::make_shared<CategorySet>(tmp);
            }
        }
    }
}

using namespace aq::nodes;
MO_REGISTER_CLASS(FaceDatabase)
