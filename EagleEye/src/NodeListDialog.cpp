#include "NodeListDialog.h"
#include "ui_nodelistdialog.h"
#include "MetaObject/MetaObjectFactory.hpp"
#include "EagleLib/Nodes/NodeInfo.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "QListWidgetItem"

NodeListDialog::NodeListDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NodeListDialog)
{
    ui->setupUi(this);
    update_slot = mo::TypedSlot<void(void)>(std::bind(&NodeListDialog::update, this));
    connection = mo::MetaObjectFactory::Instance()->ConnectConstructorAdded(&update_slot), "update", "update", update_slot.GetSignature();
    update();
}


NodeListDialog::~NodeListDialog()
{
    delete ui;
}

void NodeListDialog::update()
{
    ui->NodeList->clear();
    //auto nodes = EagleLib::NodeManager::getInstance().getConstructableNodes();
    auto nodes = mo::MetaObjectFactory::Instance()->GetConstructors(IID_NodeObject);
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        //ui->NodeList->addItem(QString::fromStdString(nodes[i]));
        //auto info  = EagleLib::NodeManager::getInstance().GetNodeInfo(nodes[i]);
        auto info = nodes[i]->GetObjectInfo();
        QTreeWidgetItem* parent = nullptr;
        if (auto node_info = dynamic_cast<EagleLib::Nodes::NodeInfo*>(info))
        {
            auto category = node_info->GetNodeCategory();
            if(category.size() == 0)
                category.push_back(node_info->GetDisplayName());
            for (int j = 0; j < ui->NodeList->topLevelItemCount(); ++j)
            {
                if (ui->NodeList->topLevelItem(j)->text(0) == QString::fromStdString(category[0]))
                {
                    parent = ui->NodeList->topLevelItem(j);
                }
            }
            if (!parent)
            {
                parent = new QTreeWidgetItem(ui->NodeList);
                ui->NodeList->addTopLevelItem(parent);
                parent->setText(0, QString::fromStdString(node_info->GetDisplayName()));
            }
            for (int k = 1; k < category.size(); ++k)
            {
                bool found = false;
                for (int j = 0; j < parent->childCount(); ++j)
                {
                    if (parent->child(j)->text(0) == QString::fromStdString(category[k]))
                    {
                        found = true;
                        parent = parent->child(j);
                        break;
                    }
                }
                if (!found)
                {
                    auto newParent = new QTreeWidgetItem(parent);
                    newParent->setText(0, QString::fromStdString(category[k]));
                    parent->addChild(newParent);
                    parent = newParent;
                }
            }
            auto node = new QTreeWidgetItem(parent);
            node->setText(0, QString::fromStdString(node_info->GetDisplayName()));
            parent->addChild(node);
        }
        else
        {
            QTreeWidgetItem* item = new QTreeWidgetItem(ui->NodeList);
            item->setText(0, QString::fromStdString(node_info->GetDisplayName()));
            ui->NodeList->addTopLevelItem(item);
        }
    }
}

void NodeListDialog::on_pushButton_clicked()
{
    if(ui->NodeList->currentItem())
    {
        add_node_signal(ui->NodeList->currentItem()->text(0).toStdString());
    }
}

void NodeListDialog::on_pushButton_2_clicked()
{
    hide();
}