#include "rccsettingsdialog.h"
#include "ui_rccsettingsdialog.h"
#include "Manager.h"
RCCSettingsDialog::RCCSettingsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RCCSettingsDialog)
{
    ui->setupUi(this);
    ui->numModules->setText(QString::number(EagleLib::NodeManager::getInstance().getNumLoadedModules()));
    EagleLib::NodeManager::getInstance().RegisterConstructorAddedCallback(boost::bind(&RCCSettingsDialog::updateDisplay, this));
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
    auto objects = EagleLib::NodeManager::getInstance().getObjectList();
    ui->linkTree->clear();
    for(auto obj : objects)
    {
        QTreeWidgetItem* objItem = new QTreeWidgetItem(ui->linkTree);
        objItem->setText(0,QString::fromStdString(obj));
        ui->linkTree->addTopLevelItem(objItem);
        auto linkDependencies = EagleLib::NodeManager::getInstance().getLinkDependencies(obj);
        for(auto link : linkDependencies)
        {
            QTreeWidgetItem* dependency = new QTreeWidgetItem(objItem);
            dependency->setText(0,QString::fromStdString(link));
            objItem->addChild(dependency);
        }
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
