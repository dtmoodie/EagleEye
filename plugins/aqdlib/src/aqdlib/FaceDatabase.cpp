#include "FaceDatabase.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <ct/reflect/print.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>

namespace aq
{
namespace nodes
{
FaceDatabase::IdentityDatabase::IdentityDatabase()
{
}

FaceDatabase::IdentityDatabase::IdentityDatabase(const std::vector<SyncedMemory>& unknown)
{
    cv::Mat desc(unknown.size(), unknown[0].getSize().width, CV_32F);
    membership.resize(unknown.size());
    for (size_t i = 0; i < unknown.size(); ++i)
    {
        unknown[i].getMatNoSync().copyTo(desc.row(i));
        identities.push_back("unknown" + std::to_string(i));
        membership[i] = i;
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
}

FaceDatabase::~FaceDatabase()
{
}

void FaceDatabase::saveUnknownFaces()
{
    mo::Mutex_t::scoped_lock lock(getMutex());
    MO_LOG(info) << "Saving " << m_unknown_crops.size() << " unknown faces to " << unknown_detections.string();
    std::ofstream ofs;
    ofs.open(unknown_detections.string() + "/unknown.db");
    if (m_unknown_crops.size())
    {
        cereal::JSONOutputArchive ar(ofs);
        IdentityDatabase unknown(m_unknown_faces);
        ar(cereal::make_nvp("unknown", unknown));
        for (size_t i = 0; i < m_unknown_crops.size(); ++i)
        {
            if (!cv::imwrite(unknown_detections.string() + '/' + std::to_string(i) + ".jpg", m_unknown_crops[i]))
            {
                MO_LOG(warning) << "Failed to write " << unknown_detections.string() + '/' + std::to_string(i) + ".jpg"
                                << " to disk";
            }
        }
    }
}

void FaceDatabase::saveKnownFaces()
{
    mo::Mutex_t::scoped_lock lock(getMutex());
    MO_LOG(info) << "Saving known faces to " << known_detections.string();
    std::ofstream ofs;
    ofs.open(known_detections.string() + "/identities.db");
    cereal::JSONOutputArchive ar(ofs);
    ar(cereal::make_nvp("face_db", m_known_faces));
}

void FaceDatabase::saveRecentFaces()
{
    mo::Mutex_t::scoped_lock lock(getMutex());
    MO_LOG(info) << "Saving " << m_recent_patches.size() << " recent faces to " << recent_detections.string();
    for (size_t i = 0; i < m_recent_patches.size(); ++i)
    {
        const auto idx = recent_detections.nextFileIndex();
        m_recent_patches[i].save(recent_detections.string(), idx, stream());
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
        MO_LOG(info) << "No identities loaded";
        m_identities = std::make_shared<CategorySet>();
    }
    else
    {
        m_identities = std::make_shared<CategorySet>(*m_identities);
    }

    m_identities->reserve(m_identities->size() + detections->size());
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
        {
            stream().waitForCompletion();
        }
    }
    if (m_recent_patches.size() > m_recent_patches.capacity() * 0.75)
    {
        saveRecentFaces();
    }

    for (const auto& det : *detections)
    {
        DetectionDescriptionPatch out_det(det);
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
            {
                stream().waitForCompletion();
            }
        }

        int match_index = -1;
        const bool euclidean = (distance_measurement.getValue() == Euclidean);
        double best_match = euclidean ? 1000.0 : 0.0;
        double mag0;
        if (!euclidean)
        {
            mag0 = cv::norm(det_desc);
        }
        if (!m_known_faces.descriptors.empty())
        {
            cv::Mat db_desc = m_known_faces.descriptors.getMatNoSync();
            std::vector<float> scores(db_desc.rows);
            for (int i = 0; i < db_desc.rows; ++i)
            {
                double dist = 0;
                if (euclidean)
                {
                    dist = cv::norm(db_desc.row(i) - det_desc);
                }
                else
                {
                    double mag1 = cv::norm(db_desc.row(i));
                    dist = det_desc.dot(db_desc.row(i)) / (mag1 * mag0);
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
                    out_det.classifications.resize(1);
                    out_det.classifications[0] = (*m_identities)[static_cast<size_t>(match_index)](best_match);
                    out_det.confidence = det.confidence;
                    out_det.id = match_index + 1;
                    sig_detectedKnownFace(*image, out_det);
                    m_recent_patches.push_back(
                        {det.aligned_patch, det.descriptor, out_det.classifications[0].cat->getName()});
                }
            }
            else
            {
                match_index = -1;
            }
        }
        if (match_index == -1)
        {
            // match to the unknown faces
            for (size_t i = 0; i < m_unknown_faces.size(); ++i)
            {
                cv::Mat desc = m_unknown_faces[i].getMatNoSync();
                double dist;
                if (euclidean)
                {
                    dist = cv::norm(desc - det_desc);
                }
                else
                {
                    auto mag1 = cv::norm(desc);
                    dist = desc.dot(det_desc) / (mag0 * mag1);
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

                m_unknown_faces.push_back(det_desc);
                auto crop_bb = det.bounding_box;
                crop_bb = crop_bb & cv::Rect2f(cv::Point2f(), host_img.size());
                m_unknown_crops.push_back(det.aligned_patch.getMat(_ctx.get()).clone());
                m_unknown_det_count.push_back(1);

                out_det.classifications.resize(1);
                out_det.classifications[0] = (*m_identities).back()();
                out_det.id = m_identities->size() - 1;

                m_recent_patches.push_back(
                    {det.aligned_patch, det.descriptor, out_det.classifications[0].cat->getName()});
            }
            else
            {
                const auto idx = match_index + m_known_faces.identities.size();
                if (idx < m_identities->size())
                {
                    out_det.classifications.resize(1);
                    out_det.classifications[0] = (*m_identities)[idx]();
                    m_recent_patches.push_back(
                        {det.aligned_patch, det.descriptor, out_det.classifications[0].cat->getName()});
                }

                ++m_unknown_det_count[match_index];
                if (m_unknown_det_count[match_index] == 10)
                {
                    if (match_index < m_unknown_faces.size())
                    {
                        auto desc = m_unknown_faces[match_index].getMatNoSync();
                        std::string base64 = cereal::base64::encode(desc.data, desc.cols * sizeof(float));
                        int idx = unknown_detections.nextFileIndex();
                        std::string stem = database_path.string() + "/unknown_" + std::to_string(idx);
                        std::ofstream ofs(stem + ".bin");
                        ofs << base64;
                        cv::imwrite(stem + ".jpg", m_unknown_crops[match_index]);
                    }
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
    IdentityDatabase load_db;
    if (boost::filesystem::exists(database_path.string() + "/identities.bin"))
    {
        std::ifstream ifs;
        ifs.open(database_path.string() + "/identities.bin");
        cereal::BinaryInputArchive ar(ifs);
        ar(cereal::make_nvp("face_db", load_db));

        loaded = true;
    }
    if (boost::filesystem::exists(database_path.string() + "/identities.db"))
    {
        MO_LOG(info) << "Loading identities from " << database_path << "/identities.db";
        std::ifstream ifs;
        ifs.open(database_path.string() + "/identities.db");
        cereal::JSONInputArchive ar(ifs);
        ar(cereal::make_nvp("face_db", load_db));
        loaded = true;
    }
    if (loaded)
    {
        mo::Mutex_t::scoped_lock lock(getMutex());
        m_known_faces = load_db;
        auto tmp = m_known_faces.identities;
        m_identities = std::make_shared<CategorySet>(tmp);
        MO_LOG(info) << "Loaded: " << *m_identities;
    }
    else
    {
        MO_LOG(warning) << "Failed to load identities from " << database_path;
    }
}

void FaceDatabase::ClassifiedPatch::save(const std::string file_path, const int index, cv::cuda::Stream& stream)
{
    const auto& p = patch.getMat(stream);
    const auto& desc = embeddings.getMat(stream);
    stream.waitForCompletion();
    {
        std::stringstream ss;

        ss << file_path << "/recent_" << classification;
        ss << "_" << std::setw(4) << std::setfill('0') << index;
        ss << ".jpg";
        if (!cv::imwrite(ss.str(), p))
        {
            MO_LOG(warning) << "Failed to write " << ss.str() << " to disk";
        }
    }
    {
        std::stringstream ss;
        ss << file_path << "/recent_" << classification;
        ss << "_" << std::setw(4) << std::setfill('0') << index;
        ss << ".bin";
        std::string base64 = cereal::base64::encode(desc.data, desc.cols * sizeof(float));
        std::ofstream ofs(ss.str());
        ofs << base64;
    }
}
}
}

using namespace aq::nodes;
MO_REGISTER_CLASS(FaceDatabase)
