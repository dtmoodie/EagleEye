#pragma once

#include <string>
#include <opencv2/core/mat.hpp>
namespace EagleLib
{
    enum CoordinateSystem
    {
        none = 0,
        image,
        cartesian,
        ECEF,
        latitude_longitude
    };
    enum Units
    {
        none = 0,
        normalized,
        homogeneous,
        pixels,
        nm,
        um,
        mm,
        cm,
        meters,
        km,
        radians,
        degrees        
    };



    template<typename T> class Sample
    {
        Sample(const T& data, double time_stamp, cv::Matx<double, 4, 4> pose, CoordinateSystem coords, Units units, const std::string& format_code, const std::string& descriptor);
        virtual ~Sample();
        operator T*() const;
        operator T&() const;
        operator T() const;
        operator double() const;


        static Sample<T> ImageSample(
            const T& data, double time_stamp = 0.0,
            cv::Matx<double, 4, 4> pose = cv::Matx<double, 4, 4>::eye(),
            CoordinateSystem coords = image, Units units = pixels,
            const std::string& format_code = "BGR",
            const std::string& descriptor = "Origin(0,0)=top_left");

        static Sample<T> Pt2dSample(
            const T& data, double time_stamp = 0.0,
            cv::Matx<double, 4, 4> pose = cv::Matx<double, 4, 4>::eye(),
            CoordinateSystem coords = image, Units units = pixels,
            const std::string& format_code = "XY",
            const std::string& descriptor = "Origin(0,0)=top_left");

        static Sample<T> Pt3dSample(
            const T& data, double time_stamp = 0.0,
            cv::Matx<double, 4, 4> pose = cv::Matx<double, 4, 4>::eye(),
            CoordinateSystem coords = image, Units units = pixels,
            const std::string& format_code = "XYZ",
            const std::string& descriptor = "Origin(0,0,0)=top_left_sensor_plane");

    protected:
        // Actual data from collection
        T _data;
        //
        double _time_stamp;
        // 4x4 homogenous transformation matrix describing the location of this data wrt to the global coordinate system
        cv::Matx<double, 4, 4> _pose;
        // Format code of the data
        std::string _format_code;
        // Describe transformation information of the camera
        std::string _descriptor;
        CoordinateSystem _coordinate_system;
        Units _units;
    };

    template<typename T> Sample<T>::Sample(const T& data, double time_stamp, cv::Matx<double, 4, 4> pose, CoordinateSystem coords, Units units, const std::string& format_code, const std::string& descriptor) :
        _data(data), _time_stamp(time_stamp), _pose(pose), _coordinate_system(coords), _units(units), _format_code(format_code), _descriptor(descriptor)
    {
    }
    template<typename T> Sample<T>::~Sample()
    {
    }
    template<typename T> Sample<T>::operator T*() const { return &_obj; }
    template<typename T> Sample<T>::operator T&() const { return _obj; }
    template<typename T> Sample<T>::operator T() const { return _obj; }
    template<typename T> Sample<T>::operator double() const { return _time_stamp; }

    template<typename T> Sample<T> Sample<T>::ImageSample(
        const T& data, double time_stamp,
        cv::Matx<double, 4, 4> pose,
        CoordinateSystem coords, Units units,
        const std::string& format_code,
        const std::string& descriptor)
    {
        return Sample<T>(data, time_stamp, pose, cords, units, format_code, descriptor);
    }

    static Sample<T> Sample<T>::Pt2dSample(
        const T& data, double time_stamp,
        cv::Matx<double, 4, 4> pose,
        CoordinateSystem coords, Units units,
        const std::string& format_code,
        const std::string& descriptor)
    {
        return Sample<T>(data, time_stamp, pose, cords, units, format_code, descriptor);
    }

    static Sample<T> Sample<T>::Pt3dSample(
        const T& data, double time_stamp,
        cv::Matx<double, 4, 4> pose,
        CoordinateSystem coords, Units units,
        const std::string& format_code,
        const std::string& descriptor)
    {
        return Sample<T>(data, time_stamp, pose, cords, units, format_code, descriptor);
    }
}