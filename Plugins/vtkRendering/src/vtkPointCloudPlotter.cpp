#include "vtkPointCloudPlotter.h"
#include <EagleLib/SyncedMemory.h>
#include <vtkFloatArray.h>
#include <vtkDataSetMapper.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <QVTKWidget2.h>
#include "Remotery.h"
#include <vtkGenericOpenGLRenderWindow.h>

using namespace EagleLib;
using namespace EagleLib::Plotting;




Plotter::PlotterType vtkPointCloudPlotterInfo::GetPlotType()
{
    return Plotter::QT_Plotter;
}
bool vtkPointCloudPlotterInfo::AcceptsParameter(Parameters::Parameter* param)
{
    auto type = param->GetTypeInfo();
    cv::Size size;
    int depth = 0;
    int channels = 0;
    if(type == Loki::TypeInfo(typeid(cv::cuda::GpuMat)))
    {
        auto typed = dynamic_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>*>(param);
        if(typed)
        {
            channels = typed->Data()->channels();
            size = typed->Data()->size();
            depth = typed->Data()->depth();
        }                    
    }else if(type == Loki::TypeInfo(typeid(cv::Mat)))
    {
        auto typed = dynamic_cast<Parameters::ITypedParameter<cv::Mat>*>(param);
        if(typed)
        {
            channels = typed->Data()->channels();
            size = typed->Data()->size();
            depth = typed->Data()->depth();
        }
    }else if(type == Loki::TypeInfo(typeid(EagleLib::SyncedMemory)))
    {
        auto typed = dynamic_cast<Parameters::ITypedParameter<EagleLib::SyncedMemory>*>(param);
        if(typed)
        {
            channels = typed->Data()->GetShape().back();
            size = typed->Data()->GetSize();
            depth = typed->Data()->GetDepth();
        }
    }

    if(depth = CV_32F)
    {
        if(size.width == 3 || size.width == 4)
        {
            return true;
        }else
        {
            if(channels == 3)
            {
                return true;
            }
        }
    }
    return false;
}
std::string vtkPointCloudPlotterInfo::GetObjectName()
{
    return "vtkPointCloudPlotter";
}
std::string vtkPointCloudPlotterInfo::GetObjectTooltip()
{
    return "Used for displaying point clouds";
}
std::string vtkPointCloudPlotterInfo::GetObjectHelp()
{
    return "Input parameter must be a point cloud in floating point tensor format or a 3 channel float image";
}

vtkPointCloudPlotterInfo g_info;

vtkPointCloudPlotter::~vtkPointCloudPlotter()
{
    Signals::thread_specific_queue::remove_from_queue(this);
    if(actor)
        renderer->RemoveActor(actor);
}
bool vtkPointCloudPlotter::AcceptsParameter(Parameters::Parameter* param)
{
    return g_info.AcceptsParameter(param);
}

void vtkPointCloudPlotter::SetInput(Parameters::Parameter* param_)
{
    if(param_)
    {
        auto type = param_->GetTypeInfo();
        if(type == Loki::TypeInfo(typeid(cv::cuda::GpuMat)))
        {
            vtkPlotter::SetInput(param_);
            _connections[&param_->update_signal] = param_->update_signal.connect(std::bind(&vtkPointCloudPlotter::OnGpuMatParameterUpdate, this, std::placeholders::_1));

        }else if(type == Loki::TypeInfo(typeid(cv::Mat)))
        {

        }else if(type == Loki::TypeInfo(typeid(EagleLib::SyncedMemory)))
        {

        }
    }    
}

void vtkPointCloudPlotter::OnMatParameterUpdate(cv::cuda::Stream* stream)
{

}



void vtkPointCloudPlotter::OnSyncedMemUpdate(cv::cuda::Stream* stream)
{

}


void vtkPointCloudPlotter::OnParameterUpdate(cv::cuda::Stream* stream)
{
    
    
    
}

std::string vtkPointCloudPlotter::PlotName() const
{
    return "vtkPointCloudPlotter";
}

void
updateCells (
    vtkSmartPointer<vtkIdTypeArray> &cells,
    vtkSmartPointer<vtkIdTypeArray> &initcells,
    vtkIdType nr_points)
{
    // If no init cells and cells has not been initialized...
    if (!cells)
        cells = vtkSmartPointer<vtkIdTypeArray>::New ();

    // If we have less values then we need to recreate the array
    if (cells->GetNumberOfTuples () < nr_points)
    {
        cells = vtkSmartPointer<vtkIdTypeArray>::New ();

        // If init cells is given, and there's enough data in it, use it
        if (initcells && initcells->GetNumberOfTuples () >= nr_points)
        {
            cells->DeepCopy (initcells);
            cells->SetNumberOfComponents (2);
            cells->SetNumberOfTuples (nr_points);
        }
        else
        {
            // If the number of tuples is still too small, we need to recreate the array
            cells->SetNumberOfComponents (2);
            cells->SetNumberOfTuples (nr_points);
            vtkIdType *cell = cells->GetPointer (0);
            for (vtkIdType i = 0; i < nr_points; ++i, cell += 2)
            {
                *cell = 1;
                *(cell+1) = i;
            }

            // Save the results in initcells
            initcells = vtkSmartPointer<vtkIdTypeArray>::New ();
            initcells->DeepCopy (cells);
        }
    }
    else
    {
        // The assumption here is that the current set of cells has more data than needed
        cells->SetNumberOfComponents (2);
        cells->SetNumberOfTuples (nr_points);
    }
}

void convertPointCloudToVTKPolyData (
    cv::InputArray cloud,
    vtkSmartPointer<vtkPolyData> &polydata,
    vtkSmartPointer<vtkIdTypeArray> &initcells, cv::cuda::Stream& stream)
{
    int num_points = 0;
    int rows, cols, type;
    if(cloud.cols() == 3)
    {
        num_points = cloud.rows();
        rows = cloud.rows();
        cols = 3;
        type = CV_32F;
    }else
    {
        if(cloud.channels() == 3)
        {
            num_points = cloud.rows() * cloud.cols();
            rows = cloud.rows();
            cols = cloud.cols();
            type = CV_32FC3;
        }        
        else
            return;
    }


    vtkSmartPointer<vtkCellArray> vertices;
    if (!polydata)
    {
        polydata = vtkSmartPointer<vtkPolyData>::New ();
        vertices = vtkSmartPointer<vtkCellArray>::New ();
        polydata->SetVerts (vertices);
    }

    // Create the supporting structures
    vertices = polydata->GetVerts ();
    if (!vertices)
        vertices = vtkSmartPointer<vtkCellArray>::New ();

    vtkIdType nr_points = num_points;
    // Create the point set
    vtkSmartPointer<vtkPoints> points = polydata->GetPoints ();
    if (!points)
    {
        points = vtkSmartPointer<vtkPoints>::New ();
        points->SetDataTypeToFloat ();
        polydata->SetPoints (points);
    }
    points->SetNumberOfPoints (nr_points);

    // Get a pointer to the beginning of the data array
    float *data = (static_cast<vtkFloatArray*> (points->GetData ()))->GetPointer (0);
    cv::Mat wrapper(rows, cols, type, data);

    if(cloud.kind() == cv::_InputArray::CUDA_GPU_MAT)
    {
        cloud.getGpuMat().download(wrapper, stream);
    }else
    {
        cloud.getMat().reshape(1, num_points).copyTo(wrapper);
    }

    vtkSmartPointer<vtkIdTypeArray> cells = vertices->GetData ();
    updateCells (cells, initcells, nr_points);

    // Set the cells and the vertices
    vertices->SetCells (nr_points, cells);
}
void
createActorFromVTKDataSet (const vtkSmartPointer<vtkDataSet> &data,
    vtkSmartPointer<vtkLODActor> &actor,
    bool use_scalars)
{
    // If actor is not initialized, initialize it here
    if (!actor)
        actor = vtkSmartPointer<vtkLODActor>::New ();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();

    mapper->SetInputData (data);

    if (use_scalars)
    {
        vtkSmartPointer<vtkDataArray> scalars = data->GetPointData ()->GetScalars ();
        double minmax[2];
        if (scalars)
        {
            scalars->GetRange (minmax);
            mapper->SetScalarRange (minmax);

            mapper->SetScalarModeToUsePointData ();
            //mapper->SetInterpolateScalarsBeforeMapping (getDefaultScalarInterpolationForDataSet (data));
            mapper->ScalarVisibilityOn ();
        }
    }
    mapper->ImmediateModeRenderingOff ();

    actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10)));
    actor->GetProperty ()->SetInterpolationToFlat ();

    /// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
    /// shown when there is a vtkActor with backface culling on present in the scene
    /// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
    // actor->GetProperty ()->BackfaceCullingOn ();

    actor->SetMapper (mapper);
    
}


void vtkPointCloudPlotter::OnGpuMatParameterUpdate(cv::cuda::Stream* stream)
{
    auto gui_thread = Signals::thread_registry::get_instance()->get_thread(Signals::GUI);
    if(Signals::get_this_thread() != gui_thread)
    {
        rmt_ScopedCPUSample(push_queue);
        Signals::thread_specific_queue::push(std::bind(&vtkPointCloudPlotter::OnGpuMatParameterUpdate, this, stream), 
            gui_thread, this);
        return;
    }
    bool shown = false;
    for(auto itr : this->render_widgets)
    {
        if(itr->isVisible())
        {
            shown = true;
            break;
        }        
    }
    if(shown == false)
        return;
    if(!param)
        return;
    auto typed = static_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>*>(param);
    if(!_opengl_vbo)
    {
        //_opengl_vbo = vtkSmartPointer<vtkMatDataBuffer>::New();
    }
    //_opengl_vbo->data_buffer.copyFrom(*typed->Data(), stream ? *stream: cv::cuda::Stream::Null());
    {
        rmt_ScopedCPUSample(transfer_point_cloud);
        bool new_actor = actor == nullptr;
        convertPointCloudToVTKPolyData(*typed->Data(), polydata, initcells, stream ? *stream : cv::cuda::Stream::Null());
        createActorFromVTKDataSet(polydata, actor, false);
        if(new_actor)
            AddAutoRemoveProp(actor);
        actor->Modified();
    }
    {
        rmt_ScopedCPUSample(render);
        RenderAll();
    }
    
}
REGISTERCLASS(vtkPointCloudPlotter, &g_info)

