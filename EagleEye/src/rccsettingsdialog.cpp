#include "rccsettingsdialog.h"
#include "ui_rccsettingsdialog.h"
#include <qfiledialog.h>
#include <boost/log/trivial.hpp>
#include <MetaObject/MetaObjectFactory.hpp>
#include <IRuntimeObjectSystem.h>
#include <FileSystemUtils.h>
RCCSettingsDialog::RCCSettingsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RCCSettingsDialog)
{
    ui->setupUi(this);
    ui->numModules->setText(QString::number(mo::MetaObjectFactory::Instance()->ListLoadedPlugins().size()));
    this->on_constructors_added = mo::TypedSlot<void(void)>(std::bind(&RCCSettingsDialog::updateDisplay, this));
    mo::MetaObjectFactory::Instance()->ConnectConstructorAdded(&this->on_constructors_added);
    ui->comboBox->addItem(RCppOptimizationLevelStrings[0]);
    ui->comboBox->addItem(RCppOptimizationLevelStrings[1]);
    ui->comboBox->addItem(RCppOptimizationLevelStrings[2]);
    ui->comboBox->addItem(RCppOptimizationLevelStrings[3]);
    //ui->comboBox->setCurrentIndex(EagleLib::ObjectManager::Instance().getOptimizationLevel());
    updateDisplay();
}
void RCCSettingsDialog::updateDisplay()
{
    
    ui->linkDirs->clear();
    ui->incDirs->clear();
    auto projects = mo::MetaObjectFactory::Instance()->ListLoadedPlugins();
    std::map<int, QTreeWidgetItem*> libDirItems;
    std::map<int, QTreeWidgetItem*> incDirItems;
    for (int i = 0; i < projects.size(); ++i)
    {
        auto libItem = new QTreeWidgetItem(ui->linkDirs);
        auto incItem = new QTreeWidgetItem(ui->incDirs);

        libItem->setText(0,QString::fromStdString(projects[i]));
        incItem->setText(0,QString::fromStdString(projects[i]));
        auto inc = mo::MetaObjectFactory::Instance()->GetObjectSystem()->GetIncludeDirList(i);
        auto lib = mo::MetaObjectFactory::Instance()->GetObjectSystem()->GetLinkDirList(i);
        for (auto dir : inc)
        {
            if (dir.Exists())
            {
                QTreeWidgetItem* dependency = new QTreeWidgetItem(incItem);
                dependency->setText(0, QString::fromStdString(dir.m_string));
                incItem->addChild(dependency);
            }
                
        }
        for (auto dir : lib)
        {
            if (dir.Exists())
            {
                QTreeWidgetItem* dependency = new QTreeWidgetItem(libItem);
                dependency->setText(0, QString::fromStdString(dir.m_string));
                libItem->addChild(dependency);
            }
        }

    }
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors();
    ui->linkTree->clear();
    ui->linkTree->setColumnCount(2);
    for(auto& constructor : constructors)
    {
        QTreeWidgetItem* objItem = new QTreeWidgetItem(ui->linkTree);
        objItem->setText(0,QString::fromStdString(constructor->GetName()));
        //objItem->setText(1, QString::number(obj.second));
        ui->linkTree->addTopLevelItem(objItem);
        /*auto linkDependencies = EagleLib::ObjectManager::Instance().getLinkDependencies(obj.first);
        for(auto& link : linkDependencies)
        {
            QTreeWidgetItem* dependency = new QTreeWidgetItem(objItem);
            dependency->setText(0,QString::fromStdString(link));
            objItem->addChild(dependency);
        }*/
    }
}

RCCSettingsDialog::~RCCSettingsDialog()
{
    delete ui;
}

void RCCSettingsDialog::on_buttonBox_accepted()
{
    //EagleLib::ObjectManager::Instance().setOptimizationLevel((RCppOptimizationLevel)ui->comboBox->currentIndex());
    //ui->comboBox->setCurrentIndex(EagleLib::ObjectManager::Instance().getOptimizationLevel());
}

void RCCSettingsDialog::on_buttonBox_rejected()
{
    //ui->comboBox->setCurrentIndex(EagleLib::ObjectManager::Instance().getOptimizationLevel());
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

    auto item = ui->incDirs->currentItem();
    int projectId = 0;
    if (item)
    {
        if (item->parent() != nullptr)
        {
            item = item->parent();
        }
        projectId = ui->incDirs->indexOfTopLevelItem(item);
    }
        
    //EagleLib::ObjectManager::Instance().addIncludeDir(dir.toStdString(), projectId);
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
    auto item = ui->linkDirs->currentItem();
    int projectId = 0;
    if (item)
    {
        while (item->parent() != nullptr)
        {
            item = item->parent();
        }
        projectId = ui->linkDirs->indexOfTopLevelItem(item);
    }
    if (projectId == -1)
    {
        LOG(warning) << "Unable to determine correct project";
        return;
    }
    //EagleLib::ObjectManager::Instance().addLinkDir(dir.toStdString(), projectId);
    ui->linkDir->clear();
    updateDisplay();
}
void RCCSettingsDialog::on_btnTestRcc_clicked()
{
    //EagleLib::ObjectManager::Instance().TestRuntimeCompilation();
}
void RCCSettingsDialog::on_btn_abort_compilation_clicked()
{
    //EagleLib::ObjectManager::Instance().abort_compilation();
}
