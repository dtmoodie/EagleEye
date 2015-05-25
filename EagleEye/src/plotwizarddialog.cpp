#include "plotwizarddialog.h"
#include "ui_plotwizarddialog.h"
#include "Manager.h"
#include "plotters/Plotter.h"

PlotWizardDialog::PlotWizardDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PlotWizardDialog)
{
    ui->setupUi(this);
    // Create plots for each plotter for demonstration purposes.
    setup();
    connect(this, SIGNAL(update(int)), this, SLOT(handleUpdate(int)));
}

void PlotWizardDialog::setup()
{
    for(int i = 0; i < previewPlots.size(); ++i)
    {
        delete previewPlots[i];
    }
    previewPlots.clear();
    auto plotters = EagleLib::PlotManager::getInstance().getAvailablePlots();
    for(int i = 0; i < plotters.size(); ++i)
    {
        shared_ptr<EagleLib::Plotter> plotter = EagleLib::PlotManager::getInstance().getPlot(plotters[i]);
        if(plotter != nullptr)
        {
            if(plotter->type() == EagleLib::Plotter::QT_Plotter)
            {
                shared_ptr<EagleLib::QtPlotter> qtPlotter(plotter);
                plotter->setCallback(boost::bind(&PlotWizardDialog::onUpdate, this, (int)previewPlots.size()));
                QCustomPlot* plot = new QCustomPlot(this);
                previewPlots.push_back(plot);

                QCPPlotTitle* title = new QCPPlotTitle(plot, QString::fromStdString(qtPlotter->plotName()));
                qtPlotter->addPlot(plot);
                plot->setInteractions((QCP::Interaction)255);
                plot->plotLayout()->insertRow(0);
                plot->plotLayout()->addElement(0,0,title);

                ui->plotPreviewLayout->addWidget(plot);
                previewPlotters.push_back(qtPlotter);
            }
        }
    }
}

PlotWizardDialog::~PlotWizardDialog()
{
    delete ui;
}
void PlotWizardDialog::onUpdate(int idx)
{
    // Emit a signal with the idx from the current processing thread to the UI thread
    emit update(idx);
}
void PlotWizardDialog::handleUpdate(int idx)
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
