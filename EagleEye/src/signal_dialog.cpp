#include "signal_dialog.h"
#include "ui_signal_dialog.h"
#include <signals/signal_manager.h>

signal_dialog::signal_dialog(Signals::signal_manager* manager, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::signal_dialog),
    _manager(manager)
{
    ui->setupUi(this);
    auto all_receivers = manager->get_receivers();
    std::map<std::string, QTreeWidgetItem*> top_level_items;
    for(auto& receiver : all_receivers)
    {
        QTreeWidgetItem* parent = nullptr;
        auto itr = top_level_items.find(receiver.signal_name);
        if(itr != top_level_items.end())
        {
            parent = itr->second;
        }else
        {
            parent = new QTreeWidgetItem(ui->treeWidget);
            parent->setText(0, QString::fromStdString(receiver.signal_name));
        }
        if(parent)
        {
            QTreeWidgetItem* child = new QTreeWidgetItem(parent);
            child->setText(1, QString::fromStdString(receiver.type.name()));
            child->setText(2, QString::fromStdString(receiver.signature.name()));
            child->setText(3, QString::fromStdString(receiver.description));
            child->setToolTip(1, QString::fromStdString(receiver.tooltip));
        }
    }
}

signal_dialog::~signal_dialog()
{
    delete ui;
}
