#include "rccsettingsdialog.h"
#include "ui_rccsettingsdialog.h"
#include "Manager.h"
RCCSettingsDialog::RCCSettingsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RCCSettingsDialog)
{
    ui->setupUi(this);
    ui->numModules->setText(QString::number(EagleLib::NodeManager::getInstance().getNumLoadedModules()));
//    RCCPPOPTIMIZATIONLEVEL_DEFAULT = 0,		// RCCPPOPTIMIZATIONLEVEL_DEBUG in DEBUG, RCCPPOPTIMIZATIONLEVEL_PERF in release. This is the default state.
//    RCCPPOPTIMIZATIONLEVEL_DEBUG,			// Low optimization, improve debug experiece. Default in DEBUG
//    RCCPPOPTIMIZATIONLEVEL_PERF,			// Optimization for performance, debug experience may suffer. Default in RELEASE
//    RCCPPOPTIMIZATIONLEVEL_NOT_SET,			// No optimization set in compile, so either underlying compiler default or set through SetAdditionalCompileOptions
//    RCCPPOPTIMIZATIONLEVEL_SIZE
    ui->comboBox->addItem("Default");
    ui->comboBox->addItem("Debug");
    ui->comboBox->addItem("Performance");
    ui->comboBox->addItem("Not Set");
    ui->comboBox->addItem("Size");
    ui->comboBox->setCurrentIndex(EagleLib::NodeManager::getInstance().getOptimizationLevel());
}

RCCSettingsDialog::~RCCSettingsDialog()
{
    delete ui;
}

void RCCSettingsDialog::on_buttonBox_accepted()
{
    EagleLib::NodeManager::getInstance().setOptimizationLevel((RCppOptimizationLevel)ui->comboBox->currentIndex());
    ui->comboBox->setCurrentIndex(EagleLib::NodeManager::getInstance().getOptimizationLevel());
}

void RCCSettingsDialog::on_buttonBox_rejected()
{
    ui->comboBox->setCurrentIndex(EagleLib::NodeManager::getInstance().getOptimizationLevel());
}

void RCCSettingsDialog::on_comboBox_currentIndexChanged(int index)
{

}
