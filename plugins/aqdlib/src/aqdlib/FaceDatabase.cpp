#include <ct/types/opencv.hpp>

#include "FaceDatabase.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>
#include <ct/reflect/print.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>

namespace aqdlib
{

    FaceDatabase::IdentityDatabase::IdentityDatabase() {}

    FaceDatabase::IdentityDatabase::IdentityDatabase(const std::vector<aq::TSyncedMemory<float>>& unknown)
    {
        const size_t descriptor_size = unknown[0].size();
        cv::Mat desc(unknown.size(), descriptor_size, CV_32F);
        membership.resize(unknown.size());
        for (size_t i = 0; i < unknown.size(); ++i)
        {
            ct::TArrayView<const float> view = unknown[i].hostAs<float>();
            ct::TArrayView<float> dst(ct::ptrCast<float>(desc.row(i).data), descriptor_size);
            view.copyTo(dst);
            identities.push_back("unknown" + std::to_string(i));
            membership[i] = i;
        }

        descriptors = aq::TSyncedImage<aq::GRAY<float>>();
    }

    /*template <class AR>
    void FaceDatabase::IdentityDatabase::save(AR& ar) const
    {
        ar(CEREAL_NVP(identities));
        cv::Mat descriptors = this->descriptors.getMat();
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
        descriptors = aq::SyncedMemory(desc);
    }

    void FaceDatabase::IdentityDatabase::save(cereal::JSONOutputArchive& ar) const
    {
        if (!descriptors.empty())
        {
            cv::Mat descriptors = this->descriptors.getMatNoSync();
            ar(cereal::make_nvp("descriptor_width", descriptors.cols));
            ar(cereal::make_nvp("num_entries", descriptors.rows));
            int count = 0;
            for (size_t i = 0; i < identities.size(); ++i)
            {
                const auto num_entries = std::count(membership.begin(), membership.end(), i);
                ar.setNextName(identities[i].c_str());
                ar.startNode();
                ar(cereal::make_size_tag(num_entries));
                while (membership[count] == i)
                {
                    ar.saveBinaryValue(descriptors.ptr<float>(count), sizeof(float) * descriptors.cols);
                    ++count;
                }
            }
        }
        else
        {
            ar(cereal::make_nvp("descriptor_width", 128));
        }
    }

    void FaceDatabase::IdentityDatabase::load(cereal::JSONInputArchive& ar)
    {
        int cols;
        int num_rows;
        ar(cereal::make_nvp("descriptor_width", cols));
        // ar(cereal::make_nvp("num_entries", num_rows));
        std::vector<cv::Mat> descs;
        // cv::Mat desc(num_rows, cols, CV_32F);

        identities.clear();
        size_t count = 0;
        membership.clear();
        int current_identity = 0;

        while (true)
        {
            const auto name = ar.getNodeName();
            if (!name)
            {
                break;
            }

            size_t num_elems = 0;
            ar.startNode();
            ar(cereal::make_size_tag(num_elems));
            cv::Mat desc(num_elems, cols, CV_32F);

            for (size_t i = 0; i < num_elems; ++i)
            {
                ar.loadBinaryValue(desc.ptr<float>(i), sizeof(float) * cols);
                membership.push_back(current_identity);
            }
            ar.finishNode();
            ++current_identity;
            count += num_elems;

            identities.push_back(name);
            descs.push_back(desc);
        }

        cv::Mat merged(count, cols, CV_32F);
        int insert = 0;
        for (cv::Mat slice : descs)
        {
            slice.copyTo(merged.rowRange(insert, insert + slice.rows));
            insert += slice.rows;
        }

        descriptors = SyncedMemory(merged);
    }*/

    FaceDatabase::~FaceDatabase() {}

    void FaceDatabase::saveUnknownFaces()
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        this->getLogger().info("Saving {} unknown faces to {}", m_unknown_crops.size(), unknown_detections.string());
        std::ofstream ofs;
        ofs.open(unknown_detections.string() + "/unknown.db");
        if (m_unknown_crops.size())
        {
            mo::JSONSaver ar(ofs);

            IdentityDatabase unknown(m_unknown_face_descriptors);
            ar(&unknown, "unknown");
            for (size_t i = 0; i < m_unknown_crops.size(); ++i)
            {
                std::string output_path = unknown_detections.string() + '/' + std::to_string(i) + ".jpg";
                if (!cv::imwrite(output_path, m_unknown_crops[i]))
                {
                    this->getLogger().warn("Failed to write {} to disk", output_path);
                }
            }
        }
    }

    void FaceDatabase::saveKnownFaces()
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        this->getLogger().info("Saving known faces to {}", known_detections.string());
        std::ofstream ofs;
        ofs.open(known_detections.string() + "/identities.db");
        mo::JSONSaver ar(ofs);
        ar(&m_known_faces, "face_db");
    }

    void FaceDatabase::saveRecentFaces()
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        auto stream = this->getStream();
        this->getLogger().info("Saving {} recent faces to {}", m_recent_patches.size(), recent_detections.string());
        for (size_t i = 0; i < m_recent_patches.size(); ++i)
        {
            const auto idx = recent_detections.nextFileIndex();
            m_recent_patches[i].save(recent_detections.string(), idx, *stream);
        }
        m_recent_patches.clear();
    }

    bool FaceDatabase::processImpl()
    {
        m_recent_patches.set_capacity(patch_buffer_size);
        if (m_known_faces.descriptors.empty() || (m_identities == nullptr))
        {
            loadDatabase();
        }
        if (m_identities == nullptr)
        {
            this->getLogger().info("No identities loaded");
            m_identities = std::make_shared<aq::CategorySet>();
        }
        else
        {
            m_identities = std::make_shared<aq::CategorySet>(*m_identities);
        }
        const uint32_t num_detections = detections->getNumEntities();
        m_identities->reserve(m_identities->size() + num_detections);
        Output_t output = *detections;
        output.setCatSet(m_identities);

        mo::IAsyncStreamPtr_t stream = this->getStream();
        cv::Mat host_img = image->getMat(stream.get());

        if (m_recent_patches.size() > m_recent_patches.capacity() * 0.75)
        {
            saveRecentFaces();
        }

        mt::Tensor<const aq::detection::AlignedPatch, 1> patches =
            detections->getComponent<aq::detection::AlignedPatch>();
        mt::Tensor<const float, 2> descriptors = detections->getComponent<aq::detection::Descriptor>();
        mt::Tensor<const aq::detection::BoundingBox2d::DType, 1> bbs =
            detections->getComponent<aq::detection::BoundingBox2d>();

        mt::Tensor<aq::detection::Classifications, 1> classifications =
            output.getComponentMutable<aq::detection::Classifications>();
        mt::Tensor<aq::detection::Id::DType, 1> ids = output.getComponentMutable<aq::detection::Id>();

        const uint32_t descriptor_size = descriptors.getShape()[1];

        for (uint32_t i = 0; i < num_detections; ++i)
        {
            mt::Tensor<const float, 1> det_desc = descriptors[i];
            cv::Mat_<float> wrapped_descriptor(descriptor_size, 1, const_cast<float*>(det_desc.data()));

            int match_index = -1;
            const bool euclidean = (distance_measurement.getValue() == Euclidean);
            double best_match = euclidean ? 1000.0 : 0.0;
            double mag0;
            if (!euclidean)
            {
                mag0 = cv::norm(wrapped_descriptor);
            }
            if (!m_known_faces.descriptors.empty())
            {
                cv::Mat database_descriptors = m_known_faces.descriptors.getMat(stream.get());
                std::vector<float> scores(database_descriptors.rows);
                for (int i = 0; i < database_descriptors.rows; ++i)
                {
                    double dist = 0;
                    if (euclidean)
                    {
                        dist = cv::norm(database_descriptors.row(i) - wrapped_descriptor);
                    }
                    else
                    {
                        double mag1 = cv::norm(database_descriptors.row(i));
                        dist = wrapped_descriptor.dot(database_descriptors.row(i)) / (mag1 * mag0);
                    }
                    scores[i] = dist;
                    if ((euclidean && (dist < best_match)) || (!euclidean && (dist > best_match)))
                    {
                        best_match = dist;
                        if (i < m_known_faces.membership.size())
                        {
                            match_index = static_cast<int>(m_known_faces.membership[i]);
                        }
                    }
                }
                if ((euclidean && (best_match < min_distance)) || (!euclidean && (best_match > min_distance)))
                {
                    if (match_index < m_identities->size())
                    {
                        aq::detection::Classifications& cls = classifications[i];
                        cls.resize(1);
                        cls[0] = (*m_identities)[static_cast<size_t>(match_index)](best_match);
                        ids[i] = match_index + 1;
                        // sig_detectedKnownFace(*image, out_det);

                        ClassifiedPatch patch{patches[i].aligned_patch,
                                              aq::TSyncedMemory<float>::copyHost(descriptors[i]),
                                              cls[0].cat->getName()};
                        m_recent_patches.push_back(std::move(patch));
                    }
                }
                else
                {
                    match_index = -1;
                }
            }
            if (match_index == -1)
            {
                auto stream = this->getStream();
                // match to the unknown faces
                for (size_t i = 0; i < m_unknown_face_descriptors.size(); ++i)
                {
                    auto desc_view = m_unknown_face_descriptors[i].host(stream.get());
                    cv::Mat_<float> desc(1, desc_view.size(), const_cast<float*>(desc_view.data()));
                    double dist;
                    if (euclidean)
                    {
                        dist = cv::norm(desc - wrapped_descriptor);
                    }
                    else
                    {
                        auto mag1 = cv::norm(desc);
                        dist = desc.dot(wrapped_descriptor) / (mag0 * mag1);
                    }

                    if ((euclidean && (dist < best_match)) || (!euclidean && (dist > best_match)))
                    {
                        best_match = dist;
                        match_index = static_cast<int>(i);
                    }
                }

                if (match_index == -1 ||
                    ((euclidean && (best_match > min_distance)) || (!euclidean && (best_match < min_distance))))
                {
                    // new unknown face
                    size_t unknown_count = m_identities->size() - m_known_faces.identities.size();
                    m_identities->push_back("unknown" + boost::lexical_cast<std::string>(unknown_count));

                    m_unknown_face_descriptors.push_back(aq::TSyncedMemory<float>::copyHost(det_desc));
                    auto crop_bb = bbs[i];
                    crop_bb = crop_bb & cv::Rect2f(cv::Point2f(), host_img.size());
                    m_unknown_crops.push_back(patches[i].aligned_patch.getMat(stream.get()).clone());
                    m_unknown_det_count.push_back(1);

                    classifications[i].resize(1);
                    classifications[i][0] = (*m_identities).back()();
                    ids[i] = m_identities->size() - 1;

                    m_recent_patches.push_back({patches[i].aligned_patch,
                                                aq::TSyncedMemory<float>::copyHost(det_desc),
                                                classifications[i][0].cat->getName()});
                }
                else
                {
                    const auto idx = match_index + m_known_faces.identities.size();
                    if (idx < m_identities->size())
                    {
                        classifications[i].resize(1);
                        classifications[i][0] = (*m_identities)[idx]();
                        m_recent_patches.push_back({patches[i].aligned_patch,
                                                    aq::TSyncedMemory<float>::copyHost(det_desc, stream),
                                                    classifications[i][0].cat->getName()});
                    }

                    ++m_unknown_det_count[match_index];
                    if (m_unknown_det_count[match_index] == 10)
                    {
                        if (match_index < m_unknown_face_descriptors.size())
                        {
                            auto desc = m_unknown_face_descriptors[match_index].host(stream.get());
                            std::string base64 = cereal::base64::encode(ct::ptrCast<unsigned char>(desc.data()),
                                                                        desc.size() * sizeof(float));
                            int idx = unknown_detections.nextFileIndex();
                            std::string stem = database_path.string() + "/unknown_" + std::to_string(idx);
                            std::ofstream ofs(stem + ".bin");
                            ofs << base64;
                            cv::imwrite(stem + ".jpg", m_unknown_crops[match_index]);
                        }
                    }
                    // sig_detectedUnknownFace(*image, out_det, m_unknown_det_count[match_index]);
                }
            }
        }
        this->output.publish(std::move(output));
        return true;
    }

    void FaceDatabase::loadDatabase()
    {
        bool loaded = false;
        IdentityDatabase load_db;
        if (boost::filesystem::exists(database_path.string() + "/identities.bin"))
        {
            std::ifstream ifs;
            ifs.open(database_path.string() + "/identities.bin");
            mo::BinaryLoader ar(ifs);
            ar(&load_db, "face_db");

            loaded = true;
        }
        if (boost::filesystem::exists(database_path.string() + "/identities.db"))
        {
            this->getLogger().info("Loading identities from {}/identities.db", database_path);
            std::ifstream ifs;
            ifs.open(database_path.string() + "/identities.db");
            mo::JSONLoader ar(ifs);
            ar(&load_db, "face_db");
            loaded = true;
        }
        if (loaded)
        {
            mo::Mutex_t::Lock_t lock(getMutex());
            m_known_faces = load_db;
            auto tmp = m_known_faces.identities;
            m_identities = std::make_shared<aq::CategorySet>(tmp);
            this->getLogger().info("Loaded: {}", *m_identities);
        }
        else
        {
            this->getLogger().warn("Failed to load identities from {}", database_path);
        }
    }

    void
    FaceDatabase::ClassifiedPatch::save(const std::string file_path, const int index, mo::IAsyncStream& stream) const
    {
        const auto patch = this->patch.getMat(&stream);
        const auto desc = this->embeddings.host(&stream);
        {
            std::stringstream ss;

            ss << file_path << "/recent_" << classification;
            ss << "_" << std::setw(4) << std::setfill('0') << index;
            ss << ".jpg";
            if (!cv::imwrite(ss.str(), patch))
            {
                MO_LOG(warn, "Failed to write {} to disk", ss.str());
            }
        }
        {
            std::stringstream ss;
            ss << file_path << "/recent_" << classification;
            ss << "_" << std::setw(4) << std::setfill('0') << index;
            ss << ".bin";

            std::string base64 =
                cereal::base64::encode(ct::ptrCast<unsigned char>(desc.data()), desc.size() * sizeof(float));
            std::ofstream ofs(ss.str());
            ofs << base64;
        }
    }

} // namespace aqdlib

using namespace aqdlib;
MO_REGISTER_CLASS(FaceDatabase)
