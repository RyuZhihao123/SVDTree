#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QStyleFactory>
#include <QFileDialog>
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    m_glWidget = new GLWidget(this);

    setWindowTitle("PtsToTree - 1RyuZhihao123(liuzhihao)");

    initWidgets();

    ui->mainToolBar->addWidget(ui->btnBackToOrigin);

    connect(ui->btnBackToOrigin,SIGNAL(clicked(bool)),this,SLOT(slot_btnBackToOrigin()));
    connect(ui->btnLoadMesh,SIGNAL(clicked(bool)),this,SLOT(slot_btnLoadModel()));
    connect(ui->btnGetTrunk,SIGNAL(clicked(bool)),this,SLOT(slot_btnGetTrunk()));
    connect(ui->btnConnectedGraph,SIGNAL(clicked(bool)),this,SLOT(slot_btnConnectGraph()));
    connect(ui->btnMinGraph,SIGNAL(clicked(bool)),this,SLOT(slot_btnMinGraph()));
    connect(ui->btnGetBins1,SIGNAL(clicked(bool)),this,SLOT(slot_btnGetBins1()));
    connect(ui->btnGetBins2,SIGNAL(clicked(bool)),this,SLOT(slot_btnGetBins2()));
    connect(ui->btnTreeSkeleton,SIGNAL(clicked(bool)),this,SLOT(slot_btnGetTreeSkeleton()));
    connect(ui->btnOptimizeSkeleton,SIGNAL(clicked(bool)),this,SLOT(slot_btnOptimizeSkeleton()));

    connect(ui->btnSaveDepthBuffer,SIGNAL(clicked(bool)),this,SLOT(slot_btnSaveDepthBuffer()));
    connect(ui->cbxDisplayMode,SIGNAL(currentIndexChanged(int)),this,SLOT(slot_cbxSetDisplayMode()));

    connect(ui->spinPointSize,SIGNAL(valueChanged(double)),this,SLOT(slot_spinChangeDisplayParameters()));
    connect(ui->spinLineWidth,SIGNAL(valueChanged(double)),this,SLOT(slot_spinChangeDisplayParameters()));

    connect(ui->ckbCompareWithOriginPts,SIGNAL(clicked(bool)),this,SLOT(slot_compareSkeletonWithOriginPts()));
    connect(ui->ckbRoaming,SIGNAL(clicked(bool)),this,SLOT(slot_startRoaming()));
    connect(ui->ckbDisplaySkeletonWithColors,SIGNAL(clicked(bool)),this,SLOT(slot_ckb_DisplaySkeletonWithColor()));

    connect(this->m_glWidget,SIGNAL(sig_CurCaptureCount(int,int)),this,SLOT(slot_updateCurCaputerCount(int,int)));
    // style
    QFile file(":/qdarkstyle/style.qss");
    if(file.open(QIODevice::ReadOnly))
    {
        QTextStream ts(&file);
        QString strStyle = ts.readAll();
        this->setStyleSheet(strStyle);
        file.close();
    }
}

void MainWindow::slot_btnLoadModel()
{
    QString filename = QFileDialog::getOpenFileName(this,"Load Point Data",".","Point File (*.xyz)");

    if(filename == "")
        return;

    QDateTime time = QDateTime::currentDateTime();
    this->m_glWidget->loadModelDataFrom(filename);
    qDebug()<<"Time of Loading:"<<time.msecsTo(QDateTime::currentDateTime())<<"ms";
}

void MainWindow::resizeEvent(QResizeEvent *e)
{
   ui->groupBox->move(this->width()-ui->groupBox->width(),0);

   this->m_glWidget->setGeometry(3,30,this->width() - ui->groupBox->width() - 10, this->height()-40);
}

void MainWindow::initWidgets()
{
    ui->spinSearchRadius->setStyle(QStyleFactory::create("Macintosh"));
}

void MainWindow::slot_btnBackToOrigin()
{
    this->m_glWidget->setToOriginPoints();
}

void MainWindow::slot_btnGetTrunk()
{
    QDateTime time = QDateTime::currentDateTime();
    m_glWidget->getTrunk(ui->spinSearchRadius->value(),false);

    qDebug()<<"Time of finding trunk:"<<time.msecsTo(QDateTime::currentDateTime())<<"ms";
}

void MainWindow::slot_btnConnectGraph()
{
       QDateTime time = QDateTime::currentDateTime();
    m_glWidget->connectGraph(ui->spinConnectInterval->value());

}

void MainWindow::slot_btnGetBins1()
{

    m_glWidget->getBins1(this->ui->spinBinsRadius->value());

}

void MainWindow::slot_btnGetBins2()
{
    m_glWidget->getBins2(this->ui->spinBinsPtsCount->value());
}

void MainWindow::slot_btnGetTreeSkeleton()
{
    m_glWidget->getTreeSkeleton();
}

void MainWindow::slot_btnMinGraph()
{
       QDateTime time = QDateTime::currentDateTime();
    m_glWidget->getMinGraph();
        qDebug()<<"Time of min value graph:"<<time.msecsTo(QDateTime::currentDateTime())<<"ms";
}

void MainWindow::slot_btnOptimizeSkeleton()
{
    m_glWidget->optimizeSkeleton(1,3);
}

void MainWindow::slot_btnSaveDepthBuffer()
{
    QString filename = QFileDialog::getExistingDirectory(this, tr("保存深度截图"),
                               ".");

    if(filename == "")
        return;

    m_glWidget->saveDepthBuffer(ui->spinCaptureCount->value(),filename);
}

void MainWindow::slot_cbxSetDisplayMode()
{
    m_glWidget->setDisplayMode((GLWidget::DISPLAY_MODE)this->ui->cbxDisplayMode->currentIndex());
}

void MainWindow::slot_spinChangeDisplayParameters()
{
    m_glWidget->setDisplayParameters(ui->spinPointSize->value(),
                                     ui->spinLineWidth->value());
}

void MainWindow::slot_compareSkeletonWithOriginPts()
{
    m_glWidget->compareSkeletonWithOriginPts(this->ui->ckbCompareWithOriginPts->isChecked());
}

void MainWindow::slot_startRoaming()
{
    m_glWidget->startRoaming(this->ui->ckbRoaming->isChecked());
}

void MainWindow::slot_updateCurCaputerCount(int a, int b)
{
    ui->lblCurCaptureCount->setText(QString("%1 / %2").arg(a).arg(b));
}

void MainWindow::slot_ckb_DisplaySkeletonWithColor()
{
    m_glWidget->displaySkeletonDepthColor(ui->ckbDisplaySkeletonWithColors->isChecked());
}

void MainWindow::closeEvent(QCloseEvent *e)
{
    QStringList list;
    list<<PATH_BRANCH_PART<<PATH_ORIGIN_POINTS<<PATH_LEAVES_PART;

    for(unsigned int i=0; i<list.size(); i++)
    {
        QFile file(list[i]);

        if(file.exists())
        {
            file.remove();
        }
    }
    e->accept();
}

MainWindow::~MainWindow()
{
    delete ui;
}
