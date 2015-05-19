#include "plotwizarddialog.h"
#include "ui_plotwizarddialog.h"

PlotWizardDialog::PlotWizardDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PlotWizardDialog)
{
    ui->setupUi(this);
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
    for(auto it = plotOptions.begin(); it != plotOptions.end(); ++it)
    {
        ui->plotOptionLayout->removeWidget((*it));
    }
    plotOptions.clear();
    availablePlotters.clear();
    availablePlotters = ParameterPlotter::getPlotters(param);
    for(auto itr = availablePlotters.begin(); itr != availablePlotters.end(); ++itr)
    {
        QCheckBox* box = new QCheckBox((*itr)->plotName());
        box->setChecked(false);
        box->setCheckable(true);
        plotOptions.push_back(box);
        ui->plotOptionLayout->addWidget(box);
    }
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
