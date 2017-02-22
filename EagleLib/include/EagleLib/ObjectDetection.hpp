#pragma once
#include <string>
#include <opencv2/core/types.hpp>
#include <vector>
#include <EagleLib/Detail/Export.hpp>
#include <cereal/cereal.hpp>
#include <Eigen/Geometry>

namespace cereal
{
    template<class AR, class T> inline
    void serialize(AR& ar, cv::Rect_<T>& rect)
    {
        ar(make_nvp("x", rect.x), make_nvp("y", rect.y), make_nvp("width", rect.width), make_nvp("height", rect.height));
    }

     // http://stackoverflow.com/questions/22884216/serializing-eigenmatrix-using-cereal-library
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m)
    {
      int32_t rows = m.rows();
      int32_t cols = m.cols();
      ar(rows);
      ar(cols);
      ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m)
    {
      int32_t rows;
      int32_t cols;
      ar(rows);
      ar(cols);

      m.resize(rows, cols);

      ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive& ar, Eigen::Transform<_Scalar, _Dim, _Mode, _Options> const& m)
    {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(rows, cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive& ar, Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& m)
    {
        int32_t rows, cols;
        ar(rows, cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }
} // namespace cereal

namespace EagleLib
{
    struct EAGLE_EXPORTS Classification
    {
        Classification(const std::string& label_ = "", float confidence_ = 0, int classNumber_ = -1);
        std::string label;
        float confidence;
        int classNumber;

        template<class AR>
        void serialize(AR& ar)
        {
            ar(CEREAL_NVP(label), CEREAL_NVP(confidence), CEREAL_NVP(classNumber));
        }
    };
    
    struct EAGLE_EXPORTS DetectedObject2d
    {
        std::vector<Classification> detections;
        cv::Rect2f boundingBox;
        long long timestamp;
        template<class AR>
        void serialize(AR& ar)
        {
            ar(CEREAL_NVP(boundingBox), CEREAL_NVP(detections), timestamp);
        }
    };

    typedef DetectedObject2d DetectedObject;

    struct EAGLE_EXPORTS DetectedObject3d
    {
        std::vector<Classification> detections;
        /*!
         * \brief pose determines the pose to the center of the object
         */
        Eigen::Affine3d pose;
        /*!
         * \brief size is the centered size of the object
         */
        Eigen::Vector3d size;
        long long timestamp;

        template<class AR>
        void serialize(AR& ar)
        {
            ar(CEREAL_NVP(pose), CEREAL_NVP(size), CEREAL_NVP(detections), CEREAL_NVP(timestamp));
        }
    };

    void EAGLE_EXPORTS CreateColormap(cv::Mat& lut, int num_classes, int ignore_class = -1);
}
