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

    ui->comboBox->addItem(RCppOptimizationLevelStrings[0]);
    ui->comboBox->addItem(RCppOptimizationLevelStrings[1]);
    ui->comboBox->addItem(RCppOptimizationLevelStrings[2]);
    ui->comboBox->addItem(RCppOptimizationLevelStrings[3]);
    //ui->comboBox->addItem(RCppOptimizationLevelStrings[4]);
    ui->comboBox->setCurrentIndex(EagleLib::NodeManager::getInstance().getOptimizationLevel());
	updateDisplay();
}
void RCCSettingsDialog::updateDisplay()
{
	ui->incDirs->setPlainText("");
	ui->linkDirs->setPlainText("");
	auto inc = EagleLib::NodeManager::getInstance().getIncludeDirs();
	auto lib = EagleLib::NodeManager::getInstance().getLinkDirs();
	for (auto dir : inc)
	{
		if (dir.size())
			ui->incDirs->appendPlainText(QString::fromStdString(dir));
	}
	for (auto dir : lib)
	{
		if (dir.size())
			ui->linkDirs->appendPlainText(QString::fromStdString(dir));
	}
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
void RCCSettingsDialog::on_btnAddIncludeDir_clicked()
{
	QString dir = ui->includeDir->text();
	if (dir.size() == 0)
	{
		dir = QFileDialog::getExistingDirectory(this, "Select include directory");
	}
	if (dir.size() == 0)
		return;
	EagleLib::NodeManager::getInstance().addIncludeDir(dir.toStdString());
	ui->includeDir->clear();
	updateDisplay();
}

void RCCSettingsDialog::on_btnAddLinkDir_clicked()
{
	QString dir = ui->linkDir->text();
	if (dir.size() == 0)
	{
		dir = QFileDialog::getExistingDirectory(this, "Select link directory");
	}
	if (dir.size() == 0)
		return;
	EagleLib::NodeManager::getInstance().addLinkDir(dir.toStdString());
	ui->linkDir->clear();
	updateDisplay();
}
