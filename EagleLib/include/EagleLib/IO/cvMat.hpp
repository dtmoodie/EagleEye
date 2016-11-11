#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <opencv2/core/mat.hpp>
#include <boost/lexical_cast.hpp>
namespace cereal
{
    void save(BinaryOutputArchive& ar, const cv::Mat& mat)
    {
        int rows, cols, type;
        bool continuous;

        rows = mat.rows;
        cols = mat.cols;
        type = mat.type();
        continuous = mat.isContinuous();

        ar & rows & cols & type & continuous;

        if (continuous) {
            const int data_size = rows * cols * mat.elemSize();
            auto mat_data = cereal::binary_data(mat.ptr(), data_size);
            ar & mat_data;
        }
        else {
            const int row_size = cols * mat.elemSize();
            for (int i = 0; i < rows; i++) {
                auto row_data = cereal::binary_data(mat.ptr(i), row_size);
                ar & row_data;
            }
        }
    }

    void load(BinaryInputArchive& ar, cv::Mat& mat)
    {
        int rows, cols, type;
        bool continuous;

        ar & rows & cols & type & continuous;

        if (continuous) {
            mat.create(rows, cols, type);
            const int data_size = rows * cols * mat.elemSize();
            auto mat_data = cereal::binary_data(mat.ptr(), data_size);
            ar & mat_data;
        }
        else {
            mat.create(rows, cols, type);
            const int row_size = cols * mat.elemSize();
            for (int i = 0; i < rows; i++) {
                auto row_data = cereal::binary_data(mat.ptr(i), row_size);
                ar & row_data;
            }
        }
    };

    void save(cereal::JSONOutputArchive& ar, cv::Mat const& mat)
    {
        int rows, cols, type;
        bool continuous;

        rows = mat.rows;
        cols = mat.cols;
        type = mat.type();
        continuous = mat.isContinuous();

        ar(CEREAL_NVP(rows));
        ar(CEREAL_NVP(cols));
        ar(CEREAL_NVP(type));
        ar(CEREAL_NVP(continuous));
        //ar & rows & cols & type & continuous;

        if (continuous) {
            const int data_size = rows * cols * mat.elemSize();
            ar.saveBinaryValue(mat.ptr(), data_size, "data");
            //auto mat_data = cereal::binary_data(mat.ptr(), data_size);
            //ar & mat_data;
        }
        else {
            const int row_size = cols * mat.elemSize();
            for (int i = 0; i < rows; i++) {
                ar.saveBinaryValue(mat.ptr(i), row_size, (std::string("data") + boost::lexical_cast<std::string>(i)).c_str());
                //auto row_data = cereal::binary_data(mat.ptr(i), row_size);
                //ar & row_data;
            }
        }
    }

    void load(cereal::JSONInputArchive& ar, cv::Mat & mat)
    {
        int rows, cols, type;
        bool continuous;

        //ar & rows & cols & type & continuous;
        ar(CEREAL_NVP(rows));
        ar(CEREAL_NVP(cols));
        ar(CEREAL_NVP(type));
        ar(CEREAL_NVP(continuous));

        if (continuous) {
            mat.create(rows, cols, type);
            const int data_size = rows * cols * mat.elemSize();
            ar.loadBinaryValue(mat.ptr(), data_size, "data");
            //auto mat_data = cereal::binary_data(mat.ptr(), data_size);
            //ar & mat_data;
        }
        else {
            mat.create(rows, cols, type);
            const int row_size = cols * mat.elemSize();
            for (int i = 0; i < rows; i++) {
                //auto row_data = cereal::binary_data(mat.ptr(i), row_size);
                //ar & row_data;
                ar.loadBinaryValue(mat.ptr(i), row_size, (std::string("data") + boost::lexical_cast<std::string>(i)).c_str());
            }
        }
    }

    template<class AR> void save(AR& ar, cv::Mat const& mat)
    {

    }
    template<class AR> void load(AR& ar, cv::Mat& mat)
    {

    }
}