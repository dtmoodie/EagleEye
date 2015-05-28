#include "Manager.h"
#include "plotters/Plotter.h"

#include "plotwizarddialog.h"
#include "ui_plotwizarddialog.h"


PlotWizardDialog::PlotWizardDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PlotWizardDialog)
{
    ui->setupUi(this);
    // Create plots for each plotter for demonstration purposes.
    setup();
    connect(this, SIGNAL(update(size_t)), this, SLOT(handleUpdate(size_t)));
}

void PlotWizardDialog::setup()
{
    for(int i = 0; i < previewPlots.size(); ++i)
    {
        delete previewPlots[i];
    }
    previewPlots.clear();
    previewPlotters.clear();
    auto plotters = EagleLib::PlotManager::getInstance().getAvailablePlots();
    for(size_t i = 0; i < plotters.size(); ++i)
    {
        shared_ptr<EagleLib::Plotter> plotter = EagleLib::PlotManager::getInstance().getPlot(plotters[i]);
        if(plotter != nullptr)
        {
            if(plotter->type() == EagleLib::Plotter::QT_Plotter)
            {
                shared_ptr<EagleLib::QtPlotter> qtPlotter(plotter);
                plotter->setCallback(boost::bind(&PlotWizardDialog::onUpdate, this, (int)previewPlots.size()));

                QWidget* plot = qtPlotter->getPlot(this);
                plot->installEventFilter(this);
                qtPlotter->addPlot(plot);
                ui->plotPreviewLayout->addWidget(plot);

                previewPlots.push_back(plot);
                previewPlotters.push_back(qtPlotter);
            }
        }
    }
    lastUpdateTime.resize(plotters.size());
}

PlotWizardDialog::~PlotWizardDialog()
{
    delete ui;
}
bool PlotWizardDialog::eventFilter(QObject *obj, QEvent *ev)
{
    if(ev->type() == QEvent::MouseButtonPress)
    {
        bool found = false;
        int i = 0;
        for(; i < previewPlots.size(); ++i)
        {
            if(previewPlots[i] == obj)
            {
                found = true;
                break;
            }
        }
        if(found == false)
            return false;
        QMouseEvent* mev = dynamic_cast<QMouseEvent*>(ev);
        if(mev->button() == Qt::MiddleButton)
        {
            currentPlotter = previewPlotters[i];
            QDrag* drag = new QDrag(obj);
            QMimeData* data = new QMimeData();
            drag->setMimeData(data);
            drag->exec();
            return true;
        }
    }
    return false;
}

void PlotWizardDialog::onUpdate(size_t idx)
{
    if(idx >= lastUpdateTime.size())
        return;
    // Emit a signal with the idx from the current processing thread to the UI thread
    // Limit the update rate by checking update time for each idx
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - lastUpdateTime[idx];
    // Prevent updating plots too often by limiting the update rate to every 30ms.
    if(delta.total_milliseconds() > 30)
    {
        lastUpdateTime[idx] = currentTime;
        emit update(idx);
    }
}
void PlotWizardDialog::handleUpdate(size_t idx)
{
    previewPlotters[idx]->doUpdate();
}

void PlotWizardDialog::plotParameter(EagleLib::Parameter::Ptr param)
{
    this->show();
    ui->tabWidget->setCurrentIndex(0);
    ui->inputDataType->setText(QString::fromStdString(TypeInfo::demangle(param->typeInfo.name())));
    for(int i = 0; i < previewPlotters.size(); ++i)
    {
        if(previewPlotters[i]->acceptsType(param))
        {
            previewPlotters[i]->setInput(param);
        }else
        {
            previewPlotters[i]->setInput();
        }
    }
}
void PlotWizardDialog::on_addPlotBtn_clicked()
{
    PlotWindow* plot = new PlotWindow(this);
    emit on_plotAdded(plot);
    connect(plot, SIGNAL(onDrop()), this, SLOT(on_drop()));
    QCheckBox* box = new QCheckBox(plot->getPlotName());
    box->setCheckable(true);
    box->setChecked(false);
    ui->currentPlots->addWidget(box);
    plotWindows.push_back(plot);
}
void PlotWizardDialog::on_drop()
{
    for(int i = 0; i < plotWindows.size(); ++i)
    {
        if(plotWindows[i] == sender())
        {
            if(currentPlotter != nullptr)
                plotWindows[i]->addPlotter(currentPlotter);
        }
    }
}

void PlotWizardDialog::on_tabWidget_currentChanged(int index)
{
    if(index == 2)
    {
        for(int i = 2; i < ui->currentPlots->rowCount(); ++i)
        {
            if(QCheckBox* box = dynamic_cast<QCheckBox*>(ui->currentPlots->itemAtPosition(i,0)))
                box->setText(plotWindows[i-2]->getPlotName());
        }
    }
}
