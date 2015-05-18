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
