#include <ct/types/opencv.hpp>

#include "FaceDatabase.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

#include <ct/reflect/print.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/imgcodecs.hpp>

#include <cereal/external/base64.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>

namespace aqdlib
{

    FaceDatabase::IdentityDatabase::IdentityDatabase() {}

    FaceDatabase::IdentityDatabase::IdentityDatabase(const std::vector<aq::TSyncedMemory<float>>& unknown,
                                                     mo::IAsyncStreamPtr_t stream)
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

        descriptors = aq::TSyncedImage<aq::GRAY<float>>(aq::SyncedImage(desc, aq::PixelFormat::kGRAY, stream));
    }

    void FaceDatabase::IdentityDatabase::load(mo::ILoadVisitor& visitor, const std::string&)
    {
        visitor(&identities, "identities");
        visitor(&membership, "membership");
        size_t descriptor_size = 0;
        visitor(&descriptor_size, "descriptor_size");
        descriptors.create(membership.size(), descriptor_size);
        cv::Mat mat = descriptors.mat();
        std::vector<ct::TArrayView<void>> arrs;
        for (size_t i = 0; i < membership.size(); ++i)
        {
            cv::Mat view = mat.row(i);
            ct::TArrayView<void> arr(ct::ptrCast<void>(view.data), descriptor_size * sizeof(float));
            arrs.push_back(arr);
        }
        visitor(&arrs, "descriptors");
    }

    void FaceDatabase::IdentityDatabase::save(mo::ISaveVisitor& visitor, const std::string&) const
    {
        visitor(&identities, "identities");
        visitor(&membership, "membership");
        const auto size = descriptors.size();
        for (const auto& mem : membership)
        {
            // MO_ASSERT_EQ(size(0), identities.size());
            MO_ASSERT_GT(identities.size(), mem);
        }

        const size_t descriptor_size = size(1);
        visitor(&descriptor_size, "descriptor_size");
        cv::Mat mat = descriptors.mat();
        std::vector<ct::TArrayView<const void>> arrs;
        for (size_t i = 0; i < membership.size(); ++i)
        {
            cv::Mat view = mat.row(i);
            ct::TArrayView<const void> arr(ct::ptrCast<const void>(view.data), descriptor_size * sizeof(float));
            arrs.push_back(arr);
        }
        visitor(&arrs, "descriptors");
    }

    FaceDatabase::~FaceDatabase()
    {
//        saveUnknownFaces();
//        saveKnownFaces();
//        saveRecentFaces();
    }

    void FaceDatabase::saveUnknownFaces()
    {
        std::vector<aq::TSyncedMemory<float>> unknown_face_descriptors;
        std::vector<aq::SyncedImage> unknown_crops;
        mo::IAsyncStreamPtr_t dst_stream = m_worker_stream;
        mo::IAsyncStreamPtr_t src_stream;

        {
            mo::Mutex_t::Lock_t lock(getMutex());
            src_stream = this->getStream();
            unknown_crops = std::move(m_unknown_crops);
            unknown_face_descriptors = std::move(m_unknown_face_descriptors);
        }

        std::string unknown_detections = this->unknown_detections.string();
        auto& logger = this->getLogger();
        logger.info("Saving {} unknown faces to {}", m_unknown_crops.size(), unknown_detections);

        auto work = [unknown_crops, unknown_face_descriptors, unknown_detections, &logger, dst_stream](
                        mo::IAsyncStream*) {
            std::ofstream ofs;
            ofs.open(unknown_detections + "/unknown.db");
            if (unknown_crops.size())
            {
                mo::JSONSaver ar(ofs);

                IdentityDatabase unknown(unknown_face_descriptors, dst_stream);
                ar(&unknown, "unknown");
                for (size_t i = 0; i < unknown_crops.size(); ++i)
                {
                    const std::string output_path = unknown_detections + '/' + std::to_string(i);
                    cv::Mat mat = unknown_crops[i].mat(dst_stream.get());
                    if (!cv::imwrite(output_path + ".jpg", mat))
                    {
                        logger.warn("Failed to write {} to disk", output_path);
                    }

                    const aq::TSyncedMemory<float>& desc = unknown_face_descriptors[i];
                    const ct::TArrayView<const float> view = desc.host(dst_stream.get());
                    std::ofstream ofs(output_path + ".bin");
                    std::string base64 =
                        cereal::base64::encode(ct::ptrCast<const uint8_t>(view.data()), view.size() * sizeof(float));
                    ofs << base64;
                }
            }
        };
        MO_ASSERT(src_stream);
        MO_ASSERT(dst_stream);
        // I don't know why this synchronize call hangs everything
        dst_stream->synchronize(*src_stream);
        dst_stream->pushWork(std::move(work));
    }

    void FaceDatabase::saveKnownFaces()
    {
        std::string dest_file_db;
        IdentityDatabase save;
        {
            mo::Mutex_t::Lock_t lock(getMutex());
            dest_file_db = known_detections.string() + "/identities.db";
            this->getLogger().info("Saving known faces to {}", dest_file_db);
            save = std::move(m_known_faces);
        }
        auto work = [save, dest_file_db](mo::IAsyncStream*)
        {
            std::ofstream ofs;
            ofs.open(dest_file_db);
            mo::JSONSaver ar(ofs);
            ar(&save, "face_db");
        };
        m_worker_stream->pushWork(std::move(work));
    }

    void FaceDatabase::saveRecentFaces()
    {
        boost::circular_buffer<ClassifiedPatch> patches;
        std::vector<std::string> save_names;
        std::vector<int> indices;
        mo::IAsyncStreamPtr_t stream;
        {
            mo::Mutex_t::Lock_t lock(getMutex());
            stream = this->getStream();
            this->getLogger().info("Saving {} recent faces to {}", m_recent_patches.size(), recent_detections.string());
            for (size_t i = 0; i < m_recent_patches.size(); ++i)
            {
                const int idx = recent_detections.nextFileIndex();
                save_names.push_back(recent_detections.string());
                indices.push_back(idx);
            }
            patches = std::move(m_recent_patches);
        }
        auto work = [patches, save_names, indices](mo::IAsyncStream* stream)
        {
            for(size_t i = 0; i < indices.size(); ++i)
            {
                patches[i].save(save_names[i], indices[i], *stream);
            }
        };
        m_worker_stream->pushWork(std::move(work));
    }

    bool FaceDatabase::matchKnownFaces(const mt::Tensor<const float, 1>& det_desc,
                                       aq::detection::Classifications& cls,
                                       aq::detection::Id::DType& id,
                                       const double mag0,
                                       const aq::SyncedImage& patch,
                                       mo::IAsyncStream& stream)
    {
        const uint32_t descriptor_size = det_desc.getShape()[0];
        if (descriptor_size == 0)
        {
            return false;
        }
        cv::Mat_<float> wrapped_descriptor(1, descriptor_size, const_cast<float*>(det_desc.data()));
        if (!m_known_faces.descriptors.empty())
        {
            const bool euclidean = (distance_measurement.getValue() == Euclidean);
            double best_match = euclidean ? 1000.0 : 0.0;
            cv::Mat database_descriptors = m_known_faces.descriptors.getMat(&stream);
            std::vector<float> scores(database_descriptors.rows);
            int32_t match_index = -1;
            for (int j = 0; j < database_descriptors.rows; ++j)
            {
                double dist = 0;
                if (euclidean)
                {
                    dist = cv::norm(database_descriptors.row(j) - wrapped_descriptor);
                }
                else
                {
                    double mag1 = cv::norm(database_descriptors.row(j));
                    dist = wrapped_descriptor.dot(database_descriptors.row(j)) / (mag1 * mag0);
                }
                scores[j] = dist;
                if ((euclidean && (dist < best_match)) || (!euclidean && (dist > best_match)))
                {
                    best_match = dist;
                    if (j < m_known_faces.membership.size())
                    {
                        match_index = static_cast<int32_t>(m_known_faces.membership[j]);
                    }
                }
            }
            if ((euclidean && (best_match < min_distance)) || (!euclidean && (best_match > min_distance)))
            {
                if (match_index < m_identities->size())
                {
                    cls.resize(1);
                    cls[0] = (*m_identities)[static_cast<size_t>(match_index)](best_match);
                    id = match_index + 1;

                    ClassifiedPatch tmp{patch, aq::TSyncedMemory<float>::copyHost(det_desc), cls[0].cat->getName()};
                    m_recent_patches.push_back(std::move(tmp));
                    return true;
                }
            }
        }
        return false;
    }

    bool FaceDatabase::matchUnknownFaces(const mt::Tensor<const float, 1>& det_desc,
                                         aq::detection::Classifications& cls,
                                         aq::detection::Id::DType& id,
                                         const double mag0,
                                         const aq::SyncedImage& patch,
                                         mo::IAsyncStream& stream)
    {
        const uint32_t descriptor_size = det_desc.getShape()[0];
        if (descriptor_size == 0)
        {
            return false;
        }
        cv::Mat_<float> wrapped_descriptor(1, descriptor_size, const_cast<float*>(det_desc.data()));
        const bool euclidean = (distance_measurement.getValue() == Euclidean);
        double best_match = euclidean ? 1000.0 : 0.0;

        // match to the unknown faces
        int32_t match_index = -1;
        for (size_t i = 0; i < m_unknown_face_descriptors.size(); ++i)
        {
            auto desc_view = m_unknown_face_descriptors[i].host(&stream);
            cv::Mat_<float> desc(1, desc_view.size(), const_cast<float*>(desc_view.data()));
            double dist;
            if (euclidean)
            {
                dist = cv::norm(desc - wrapped_descriptor);
                if (dist < best_match)
                {
                    best_match = dist;
                    match_index = static_cast<int32_t>(i);
                }
            }
            else
            {
                auto mag1 = cv::norm(desc);
                dist = desc.dot(wrapped_descriptor) / (mag0 * mag1);
                if (dist > best_match)
                {
                    best_match = dist;
                    match_index = static_cast<int32_t>(i);
                }
            }
        }

        if (match_index == -1 ||
            ((euclidean && (best_match > min_distance)) || (!euclidean && (best_match < min_distance))))
        {
            onNewUnknownFace(det_desc, cls, id, patch);
            return false;
        }
        else
        {
            const auto idx = match_index + m_known_faces.identities.size();
           if (idx < m_identities->size())
            {
                cls.resize(1);
                cls[0] = (*m_identities)[idx]();
                ClassifiedPatch tmp{patch, aq::TSyncedMemory<float>::copyHost(det_desc), cls[0].cat->getName()};
                m_recent_patches.push_back(std::move(tmp));
            }

            ++m_unknown_det_count[match_index];
            if (m_unknown_det_count[match_index] == 10)
            {
                saveUnknownFace(match_index);
            }
        }
        return true;
    }

    void FaceDatabase::saveUnknownFace(uint32_t match_index)
    {
        if (match_index < m_unknown_face_descriptors.size())
        {
            auto stream = this->getStream();
            const auto& descriptor = m_unknown_face_descriptors[match_index];
            const ct::TArrayView<const float> view = descriptor.host(stream.get());
            std::string base64 =
                cereal::base64::encode(ct::ptrCast<const uint8_t>(view.data()), view.size() * sizeof(float));
            int file_idx = unknown_detections.nextFileIndex();
            std::string stem = database_path.string() + "/unknown_" + std::to_string(file_idx);
            std::ofstream ofs(stem + ".bin");
            ofs << base64;
            cv::imwrite(stem + ".jpg", m_unknown_crops[match_index].mat());
        }
    }

    void FaceDatabase::onNewUnknownFace(const mt::Tensor<const float, 1>& det_desc,
                                        aq::detection::Classifications& cls,
                                        aq::detection::Id::DType& id,
                                        const aq::SyncedImage& patch)
    {
        // new unknown face
        size_t unknown_count = m_identities->size() - m_known_faces.identities.size();
        std::string name = "unknown" + boost::lexical_cast<std::string>(unknown_count);
        m_identities->push_back(name);
        mo::IAsyncStream::Ptr_t stream = this->getStream();
        m_unknown_face_descriptors.push_back(aq::TSyncedMemory<float>::copyHost(det_desc, stream));

        m_unknown_crops.push_back(patch);
        m_unknown_det_count.push_back(1);

        cls.resize(1);
        cls[0] = (*m_identities).back()();
        id = m_identities->size() - 1;

        m_recent_patches.push_back({patch, aq::TSyncedMemory<float>::copyHost(det_desc, stream), name});
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

        mt::Tensor<aq::detection::Classifications, 1> classifications =
            output.getComponentMutable<aq::detection::Classifications>();

        mt::Tensor<aq::detection::Id::DType, 1> ids = output.getComponentMutable<aq::detection::Id>();

        for (uint32_t i = 0; i < num_detections; ++i)
        {
            mt::Tensor<const float, 1> det_desc = descriptors[i];
            cv::Mat_<float> wrapped_descriptor(1, det_desc.getShape()[0], const_cast<float*>(det_desc.data()));
            const bool euclidean = (distance_measurement.getValue() == Euclidean);
            double mag0;
            if (!euclidean)
            {
                mag0 = cv::norm(wrapped_descriptor);
            }

            if (matchKnownFaces(det_desc, classifications[i], ids[i], mag0, patches[i].aligned_patch, *stream))
            {
                continue;
            }

            if (matchUnknownFaces(det_desc, classifications[i], ids[i], mag0, patches[i].aligned_patch, *stream))
            {
                continue;
            }
        }
        this->output.publish(std::move(output), mo::tags::param = &this->detections_param);
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
            m_known_faces = std::move(load_db);
            auto tmp = m_known_faces.identities;
            m_identities = std::make_shared<aq::CategorySet>(tmp);
            this->getLogger().info("Loaded: {}", *m_identities);
        }
        else
        {
            this->getLogger().warn("Failed to load identities from {}", database_path);
        }
    }

    void FaceDatabase::nodeInit(bool)
    {
        m_worker_stream = m_worker_thread.asyncStream();
        MO_ASSERT(m_worker_stream);
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
