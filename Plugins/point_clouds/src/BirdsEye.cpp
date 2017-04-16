#ifdef HAVE_PCL
#include "BirdsEye.hpp"
#include "Aquila/Nodes/NodeInfo.hpp"

using namespace aq::Nodes;

bool BirdsEye::ProcessImpl()
{
    if(*point_cloud)
    {
        std::vector<cv::Mat> slices;
        slices.resize(this->slices + 2);
        for(int i = 0; i < slices.size(); ++i)
        {
            slices[i].create(height, width, CV_32F);
            slices[i].setTo(0);
        }
        const pcl::PointCloud<pcl::PointXYZI>& pc = *(*point_cloud);
        float height_stride = (max_z - min_z) / float(this->slices);
        cv::Mat cell_count(height, width, CV_32F);
        cell_count.setTo(0);
        for(const pcl::PointXYZI& pt : pc)
        {
            int x = pt.x / resolution;
            int y = pt.y / resolution;
            x -= width / 2;
            y -= height / 2;
            if(x >= 0 && x < width && y >= 0 && y < height)
            {
                if(pt.z >= min_z && pt.z <= max_z)
                {
                    int height_idx = (pt.z - min_z) / height_stride;
                    if(pt.z > slices[height_idx].at<float>(y, x))
                    {
                        slices[height_idx].at<float>(y, x) = pt.z;
                        slices[this->slices].at<float>(y, x) = pt.intensity;
                    }
                    cell_count.at<float>(y,x) += 1;
                }
            }
        }
        for(int i = 0; i < height; ++i)
        {
            for(int j = 0; j < width; ++j)
            {
                slices[this->slices + 1].at<float>(i,j) =
                        std::min<float>(1.0f,log(cell_count.at<float>(i,j) + 1) / log(64.0f));
            }
        }
        birds_eye_view_param.UpdateData(slices, point_cloud_param.GetTimestamp());
    }
    return true;
}

MO_REGISTER_CLASS(BirdsEye)
#endif
