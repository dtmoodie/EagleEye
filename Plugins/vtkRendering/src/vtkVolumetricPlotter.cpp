#include "vtkVolumetricPlotter.h"
#include <EagleLib/SyncedMemory.h>
#include <EagleLib/utilities/ColorMapperFactory.hpp>
#include <EagleLib/utilities/IColorMapper.hpp>
#include <QVTKInteractor.h>
#include <QVTKWidget2.h>
#include <vtkBox.h>
#include <vtkBoxWidget.h>
#include <vtkClipVolume.h>
#include <vtkColorTransferFunction.h>
#include <vtkDataReader.h>
#include <vtkErrorCode.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPlanes.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkStructuredPoints.h>
#include <vtkUnsignedShortArray.h>
#include <vtkVolumeProperty.h>

using namespace EagleLib;
using namespace EagleLib::Plotting;

class vtkBoxWidgetCallback : public vtkCommand
{
  public:
    static vtkBoxWidgetCallback* New() { return new vtkBoxWidgetCallback; }
    virtual void Execute(vtkObject* caller, unsigned long, void* ptr)
    {
        vtkBoxWidget* widget = reinterpret_cast<vtkBoxWidget*>(caller);
        if (this->Mapper)
        {
            vtkPlanes* planes = vtkPlanes::New();
            widget->GetPlanes(planes);
            this->Mapper->SetClippingPlanes(planes);
            planes->Delete();
        }
    }
    void SetMapper(vtkSmartVolumeMapper* m) { this->Mapper = m; }

  protected:
    vtkBoxWidgetCallback() { this->Mapper = 0; }

    vtkSmartVolumeMapper* Mapper;
};

bool vtkVolumetricPlotter::AcceptsParameter(mo::IParameter* param)
{
    auto type = param->GetTypeInfo();
    if (type == mo::TypeInfo(typeid(TS<SyncedMemory>)))
    {
        auto typed = dynamic_cast<mo::ITypedParameter<TS<SyncedMemory>>*>(param);
        if (typed)
        {
            auto ts_data = typed->GetDataPtr();
            if (ts_data)
            {
                auto shape = ts_data->GetShape();
                if (shape.size() == 4)
                {
                    if (shape[0] == 1 /*channels are packed in last dim */ ||
                        shape[3] == 1 /*channels are packed in first dim */)
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

vtkVolumetricPlotter::vtkVolumetricPlotter()
{
    _callback = nullptr;
}

vtkVolumetricPlotter::~vtkVolumetricPlotter()
{
    if (_clipping_function && _callback)
        _clipping_function->RemoveObserver(_callback);
    if (_mapper)
    {
        _mapper->RemoveAllClippingPlanes();
    }
}

void vtkVolumetricPlotter::SetInput(mo::IParameter* param_)
{
    if (param_)
    {
        auto type = param_->GetTypeInfo();
        if (type == mo::TypeInfo(typeid(TS<SyncedMemory>)))
        {
            vtkPlotterBase::SetInput(param_);
            //_connections[&param_->update_signal] =
            //param_->update_signal.connect(std::bind(&vtkVolumetricPlotter::OnSyncedMemUpdate, this,
            //std::placeholders::_1));
            auto typed = dynamic_cast<mo::ITypedParameter<TS<SyncedMemory>>*>(param_);
            if (typed && typed->GetDataPtr())
            {
                // vtkSmartPointer<vtkSmartVolumeMapper> volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
                auto data = typed->GetDataPtr();
                auto shape = data->GetShape();
                int extent[6];
                int dim_count = 0;

                for (int i = 0; i < shape.size(); ++i)
                {
                    if (shape[i] != 1)
                    {
                        extent[2 * dim_count] = 0;
                        extent[2 * dim_count + 1] = shape[i];
                        ++dim_count;
                    }
                }
                size_t size = extent[1] * extent[3] * extent[5];
                renderer->SetBackground(0.1, 0.4, 0.2);
                vtkSmartPointer<vtkStructuredPoints> dataset = vtkSmartPointer<vtkStructuredPoints>::New();
                dataset->SetExtent(extent);
                dataset->SetOrigin(0, 0, 0);
                dataset->SetSpacing(1, 1, 1);
                auto points = dataset->GetPointData();
                vtkUnsignedShortArray* scalars = vtkUnsignedShortArray::New();

                scalars->SetNumberOfComponents(1);
                scalars->SetNumberOfValues(size);
                size_t channel_stride = extent[3] * extent[5];
                if (shape[0] != 1 && shape[3] == 1)
                {
                    for (int c = 0; c < shape[0]; ++c)
                    {
                        cv::Mat wrapped(extent[3], extent[5], CV_16U, scalars->GetPointer(c * channel_stride));
                        cv::Mat input = data->GetMat(cv::cuda::Stream::Null(), c);
                        input.convertTo(wrapped, CV_16U);
                    }
                }
                scalars->DataChanged();
                points->SetScalars(scalars);

                _volume = vtkSmartPointer<vtkVolume>::New();

                _mapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
                _mapper->SetBlendModeToComposite(); // composite first
                _clipping_function = vtkSmartPointer<vtkBoxWidget>::New();
                for (auto widget : render_widgets)
                {
                    auto interactor = widget->GetInteractor();
                    _clipping_function->SetInteractor(interactor);
                }
                _clipping_function->SetPlaceFactor(1.01);
                _clipping_function->SetInputData(dataset);
                _clipping_function->SetInsideOut(true);
                _clipping_function->PlaceWidget();

                _callback = vtkBoxWidgetCallback::New();
                _callback->SetMapper(_mapper);
                _clipping_function->AddObserver(vtkCommand::InteractionEvent, _callback);

                _clipping_function->EnabledOn();
                _clipping_function->GetSelectedFaceProperty()->SetOpacity(0.0);
                _mapper->SetInputData(dataset);
                _volume->SetMapper(_mapper);

                _volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
                _volumeProperty->ShadeOff();
                _volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

                _compositeOpacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
                _compositeOpacity->AddPoint(0.0, 0.0);
                _compositeOpacity->AddPoint(80.0, 1.0);
                _compositeOpacity->AddPoint(80.1, 1.0);
                _compositeOpacity->AddPoint(255.0, 1.0);
                _volumeProperty->SetScalarOpacity(_compositeOpacity); // composite first.

                _color = vtkSmartPointer<vtkColorTransferFunction>::New();
                auto scheme = ColorMapperFactory::Instance()->Create(
                    colormapping_scheme.enumerations[colormapping_scheme.currentSelection],
                    opacity_max_value - opacity_min_value,
                    opacity_min_value);
                if (scheme)
                {
                    cv::Mat_<float> lut = scheme->GetMat(0, opacity_max_value, 50);
                    for (int i = 0; i < lut.rows; ++i)
                    {
                        _color->AddRGBPoint(lut(i, 0), lut(i, 1), lut(i, 2), lut(i, 3));
                    }
                    _volumeProperty->SetColor(_color);
                }
                else
                {
                    _color->AddRGBPoint(0, 0.0, 0.0, 1.0);
                    _color->AddRGBPoint(1024, 1.0, 0.0, 0.0);
                    _color->AddRGBPoint(2048, 1.0, 1.0, 1.0);
                }
                _volumeProperty->SetColor(_color);

                _volume->SetProperty(_volumeProperty);

                AddAutoRemoveProp(_volume);
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
void vtkVolumetricPlotter::PlotInit(bool firstInit)
{
    vtkPlotterBase::PlotInit(firstInit);
    colormapping_scheme.enumerations = ColorMapperFactory::Instance()->ListSchemes();
    colormapping_scheme.values.clear();
    for (int i = 0; i < colormapping_scheme.enumerations.size(); ++i)
        colormapping_scheme.values.push_back(i);
    if (firstInit)
        colormapping_scheme.currentSelection = 0;
}
void vtkVolumetricPlotter::onUpdate(mo::IParameter* param, cv::cuda::Stream* stream)
{
    // vtkPlotterBase::onUpdate(param, stream);
    if (param == &opacity_max_value_param || param == &opacity_sharpness_param || param == &opacity_min_value_param)
    {
        _compositeOpacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
        _compositeOpacity->AddPoint(opacity_min_value, 0.0);
        _compositeOpacity->AddPoint(opacity_max_value, 1.0);
        _volumeProperty->SetScalarOpacity(_compositeOpacity); // composite first.
    }
    if (param == &colormapping_scheme_param)
    {
        auto scheme = ColorMapperFactory::Instance()->Create(
            colormapping_scheme.enumerations[colormapping_scheme.currentSelection],
            opacity_max_value - opacity_min_value,
            opacity_min_value);
        if (scheme)
        {
            cv::Mat_<float> lut = scheme->GetMat(0, opacity_max_value, 50);
            _color = vtkSmartPointer<vtkColorTransferFunction>::New();
            for (int i = 0; i < lut.rows; ++i)
            {
                _color->AddRGBPoint(lut(i, 0), lut(i, 1), lut(i, 2), lut(i, 3));
            }
            _volumeProperty->SetColor(_color);
        }
    }
    mo::ThreadSpecificQueue::Push(
        [this]() -> void {
            for (auto itr : this->render_widgets)
            {
                itr->GetRenderWindow()->Render();
            }
        },
        mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI),
        this);
}

MO_REGISTER_CLASS(vtkVolumetricPlotter)
