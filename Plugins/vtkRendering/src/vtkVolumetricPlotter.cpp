#include "vtkVolumetricPlotter.h"
#include <EagleLib/SyncedMemory.h>
#include <vtkDataReader.h>
#include <vtkStructuredPoints.h>
#include <vtkErrorCode.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkPointData.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkUnsignedShortArray.h>
#include <vtkPiecewiseFunction.h>

using namespace EagleLib;
using namespace EagleLib::Plotting;

bool vtkVolumetricPlotterInfo::AcceptsParameter(Parameters::Parameter* param)
{
    auto type = param->GetTypeInfo();
    if(type == Loki::TypeInfo(typeid(TS<SyncedMemory>)))
    {
        auto typed = dynamic_cast<Parameters::ITypedParameter<TS<SyncedMemory>>*>(param);
        if(typed)
        {
            auto ts_data = typed->Data();
            if(ts_data)
            {
                auto shape = ts_data->GetShape();
                if(shape.size() == 4)
                {
                    if(shape[0] == 1 /*channels are packed in last dim */ || shape[3] == 1 /*channels are packed in first dim */)
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

std::string vtkVolumetricPlotterInfo::GetObjectName()
{
    return "vtkVolumetricPlotter";
}

std::string vtkVolumetricPlotterInfo::GetObjectTooltip()
{
    return "Plot data as if each channel is along the Z axis";
}

std::string vtkVolumetricPlotterInfo::GetObjectHelp()
{
    return "Volumetric plot .....";
}

static vtkVolumetricPlotterInfo g_volumeInfo;


vtkVolumetricPlotter::~vtkVolumetricPlotter()
{
}
bool vtkVolumetricPlotter::AcceptsParameter(Parameters::Parameter* param)
{
    return g_volumeInfo.AcceptsParameter(param);
}
void vtkVolumetricPlotter::SetInput(Parameters::Parameter* param_)
{
    if(param_)
    {
        auto type = param_->GetTypeInfo();
        if(type == Loki::TypeInfo(typeid(TS<SyncedMemory>)))
        {
            vtkPlotter::SetInput(param_);
            _connections[&param_->update_signal] = param_->update_signal.connect(std::bind(&vtkVolumetricPlotter::OnSyncedMemUpdate, this, std::placeholders::_1));
            auto typed = dynamic_cast<Parameters::ITypedParameter<TS<SyncedMemory>>*>(param_);
            if(typed && typed->Data())
            {
                //vtkSmartPointer<vtkSmartVolumeMapper> volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
                auto data = typed->Data();
                auto shape = data->GetShape();
                int extent[6];
                int dim_count = 0;
                
                for(int i = 0; i < shape.size(); ++i)
                {
                    if(shape[i] != 1)
                    {
                        extent[2*dim_count] = 0;
                        extent[2*dim_count+1] = shape[i];
                        ++dim_count;
                    }
                }
                size_t size = extent[1] * extent[3] * extent[5];
                renderer->SetBackground(0.1,0.4,0.2);
                vtkSmartPointer<vtkStructuredPoints> dataset = vtkSmartPointer<vtkStructuredPoints>::New();
                dataset->SetExtent(extent);
                dataset->SetOrigin(0,0,0);
                dataset->SetSpacing(1,1,1);
                auto points = dataset->GetPointData();
                vtkUnsignedShortArray *scalars = vtkUnsignedShortArray::New();
                
                scalars->SetNumberOfComponents(1);
                scalars->SetNumberOfValues(size);
                size_t channel_stride = extent[3]*extent[5];
                if(shape[0] != 1 && shape[3] == 1)
                {
                    for(int c = 0; c < shape[0]; ++c)
                    {
                        cv::Mat wrapped(extent[3], extent[5], CV_16U, scalars->GetPointer(c*channel_stride));
                        cv::Mat input = data->GetMat(cv::cuda::Stream::Null(), c);
                        input.convertTo(wrapped, CV_16U);
                    }
                }
                scalars->DataChanged();
                points->SetScalars(scalars);
                
                vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();

                    vtkSmartPointer<vtkSmartVolumeMapper> mapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
                        mapper->SetBlendModeToComposite(); // composite first
                        mapper->SetInputData(dataset);
                        volume->SetMapper(mapper);

                    vtkSmartPointer<vtkVolumeProperty> volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
                        volumeProperty->ShadeOff();
                        volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

                        vtkSmartPointer<vtkPiecewiseFunction> compositeOpacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
                            compositeOpacity->AddPoint(0.0,0.0);
                            compositeOpacity->AddPoint(80.0,1.0);
                            compositeOpacity->AddPoint(80.1,1.0);
                            compositeOpacity->AddPoint(255.0,1.0);
                            volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

                        vtkSmartPointer<vtkColorTransferFunction> color =  vtkSmartPointer<vtkColorTransferFunction>::New();
                            color->AddRGBPoint(0,    0.0, 0.0, 1.0);
                            color->AddRGBPoint(1024, 1.0, 0.0, 0.0);
                            color->AddRGBPoint(2048, 1.0, 1.0, 1.0);
                            volumeProperty->SetColor(color);

                        volume->SetProperty(volumeProperty);

                renderer->AddViewProp(volume);
            }
        }
    }
}
void vtkVolumetricPlotter::OnParameterUpdate(cv::cuda::Stream* stream)
{
}
void vtkVolumetricPlotter::OnMatParameterUpdate(cv::cuda::Stream* stream)
{
}
void vtkVolumetricPlotter::OnGpuMatParameterUpdate(cv::cuda::Stream* stream)
{
}
void vtkVolumetricPlotter::OnSyncedMemUpdate(cv::cuda::Stream* stream)
{
}
std::string vtkVolumetricPlotter::PlotName() const
{
    return "vtkVolumetricPlotter";
}

REGISTERCLASS(vtkVolumetricPlotter, &g_volumeInfo)

