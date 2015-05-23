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
}

void PlotWizardDialog::setup()
{
    for(int i = ui->plotPreviewLayout->count() - 1; i > 0 ; --i)
    {
        QCustomPlot* plot = dynamic_cast<QCustomPlot*>(ui->plotPreviewLayout->itemAt(i));
        if(plot)
        {
            delete plot;
        }
    }
    auto plotters = EagleLib::PlotManager::getInstance().getAvailablePlots();
    for(int i = 0; i < plotters.size(); ++i)
    {
        shared_ptr<EagleLib::Plotter> plotter = EagleLib::PlotManager::getInstance().getPlot(plotters[i]);
        if(plotter != nullptr)
        {
            if(plotter->type() == EagleLib::Plotter::QT_Plotter)
            {
                shared_ptr<EagleLib::QtPlotter> qtPlotter(plotter);
                QCustomPlot* plot = new QCustomPlot(this);
                QCPPlotTitle* title = new QCPPlotTitle(plot, QString::fromStdString(qtPlotter->plotName()));
                qtPlotter->addPlot(plot);
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
void PlotWizardDialog::plotParameter(EagleLib::Parameter::Ptr param)
{
    this->show();
    ui->tabWidget->setCurrentIndex(0);
    ui->inputDataType->setText(QString::fromStdString(TypeInfo::demangle(param->typeInfo.name())));
//    for(auto it = plotOptions.begin(); it != plotOptions.end(); ++it)
//    {
//        //ui->plotOptionLayout->removeWidget((*it));
//    }
//    plotOptions.clear();
//    availablePlotters.clear();
//    availablePlotters = ParameterPlotter::getPlotters(param);
//    for(auto itr = availablePlotters.begin(); itr != availablePlotters.end(); ++itr)
//    {
//        QCheckBox* box = new QCheckBox((*itr)->plotName());
//        box->setChecked(false);
//        box->setCheckable(true);
//        plotOptions.push_back(box);
//        //ui->plotOptionLayout->addWidget(box);
//    }
}
void PlotWizardDialog::on_addPlotBtn_clicked()
{
    PlotWindow* plot = new PlotWindow(this);
    emit on_plotAdded(plot);
    QCheckBox* box = new QCheckBox(plot->getPlotName());
    box->setCheckable(true);
    box->setChecked(false);
    ui->currentPlots->addWidget(box);
    plotWindows.push_back(plot);
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
