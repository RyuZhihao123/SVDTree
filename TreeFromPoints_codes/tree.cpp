#include "tree.h"

typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double> > ,PointCloud<double>,3 /* dim */>
        KDTree;

Tree::Tree() :m_pointVBO_Vertex_Count(0),m_graphVBO_Vertex_Count(0),
    m_rootMinGraph(NULL),m_binRoot(NULL),m_process_stage(_STAGE_1),
    m_isSkeletonBeColored(false){}

void Tree::loadPointsFrom(QVector<QVector3D> &vec, const QString &filename)
{
    vec.clear();
    QFile file(filename);

    if(!file.open(QIODevice::ReadOnly))
        return;

    QDataStream ds(&file);

    while(!ds.atEnd())
    {
        float x,y,z;
        ds>>x>>y>>z;
        vec.push_back(QVector3D(x,y,z));
    }
    file.close();
}

void Tree::clearAllBuffers()
{
    QStringList filelist;
    filelist<<PATH_BRANCH_PART<<PATH_LEAVES_PART<<PATH_ORIGIN_POINTS;

    for(unsigned int i=0; i<filelist.size(); i++)
    {
        QFile file(filelist[i]);
        if (file.exists())
        {
            file.remove();
        }
    }
}

void Tree::setToOriginPoints()
{
    enterCurrentStageMode(_STAGE_1);
    loadPointsFrom(this->m_vertexs,PATH_ORIGIN_POINTS);
    createPointVBO(this->m_vertexs);
}

void Tree::VectorToPointCloud(const QVector<QVector3D> &vec, PointCloud<double> &cloud)
{
    cloud.pts.resize(vec.size());
    for(int i=0; i<vec.size(); i++)
    {
        cloud.pts[i].x = vec[i].x();
        cloud.pts[i].y = vec[i].y();
        cloud.pts[i].z = vec[i].z();
    }
}

int Tree::getTreeCount(TreeNode *&root) const
{
    QVector<TreeNode*> branchs;
    branchs.push_back(root);

    int count =0;
    while(branchs.size()>0)
    {
        TreeNode* cur = branchs.front();
        branchs.pop_front();
        count++;

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            branchs.push_back(cur->childs[i]);
        }
    }
    return count;
}

void Tree::savePointsTo(QVector<QVector3D> &vec, const QString &filename)
{
    QFile file(filename);

    if(!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return;

    QDataStream ds(&file);

    for(unsigned int i=0 ;i <vec.size(); i++)
    {
        ds<<vec[i].x()<<vec[i].y()<<vec[i].z();
    }
    file.close();
}

void Tree::clearMinTree()
{
    if(!this->m_rootMinGraph)
        return;

    QVector<TreeNode*> stack;
    stack.push_back(this->m_rootMinGraph);
    QVector<TreeNode*> removeList;

    while(stack.size()!=0)
    {
        TreeNode* cur = stack.front();
        stack.pop_front();
        removeList.push_back(cur);

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            stack.push_back(cur->childs[i]);
        }
    }

    for(int i=removeList.size()-1; i>=0; i--)
    {
        removeList[i]->childs.clear();
        delete removeList[i];
    }

    m_rootMinGraph =  NULL;
}

void Tree::clearBinTree()
{
    if(!this->m_binRoot)
        return;

    QVector<Bin*> stack;
    stack.push_back(this->m_binRoot);
    QVector<Bin*> removeList;

    while(stack.size()!=0)
    {
        Bin* cur = stack.front();
        stack.pop_front();
        removeList.push_back(cur);

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            stack.push_back(cur->childs[i]);
        }
    }

    for(int i=removeList.size()-1; i>=0; i--)
    {
        removeList[i]->childs.clear();
        removeList[i]->pts.clear();
        removeList[i]->ptsId.clear();

        delete removeList[i];
    }

    m_binRoot =  NULL;
}

void Tree::enterCurrentStageMode(PROCESS_STAGE stage)
{
    this->m_process_stage = stage;

    if(this->m_process_stage == _STAGE_1)  // 刚加载数据
    {
        this->m_vertexs.clear();
        this->m_curVertexes.clear();
        this->m_leafPart.clear();
        this->m_branchGraph.clear();

        this->clearBinTree();
        this->clearMinTree();

        this->createGraphVBO();

        // buffer
        QStringList filelist;
        filelist<<PATH_BRANCH_PART<<PATH_LEAVES_PART;

        for(unsigned int i=0; i<filelist.size(); i++)
        {
            QFile file(filelist[i]);
            if (file.exists())
            {
                file.remove();
            }
        }
    }
    if(this->m_process_stage == _STAGE_2)
    {
        this->m_curVertexes.clear();
        this->m_leafPart.clear();
        this->m_branchGraph.clear();

        this->clearBinTree();
        this->clearBinTree();
        this->createGraphVBO();
    }
    if(this->m_process_stage != _STAGE_7)
    {
        this->m_skeletonPts.clear();
    }
}

void Tree::comperaSkeletonWithOriginPts(bool b)
{
    if(this->m_process_stage == _STAGE_7)
    {
        if(b)
        {
            QVector<QVector3D> vec;
            loadPointsFrom(vec,PATH_ORIGIN_POINTS);
            createPointVBO(vec);
        }
        else
        {
            createPointVBO(this->m_skeletonPts);
        }
    }
}


QVector3D Tree::getCenterFrom(QVector<QVector3D> &vertices)
{
    if(vertices.size() == 0)
        return QVector3D(0,0,0);

    int miny = 9999999;
    int maxy = -999999;
    for(int i=0; i<vertices.size(); i++)
    {
        if(miny > vertices[i].y())
            miny = vertices[i].y();
        if(maxy < vertices[i].y())
            maxy = vertices[i].y();
    }

    return QVector3D(0,(maxy+miny)/2,0);
}

int Tree::locateRootindex(const QVector<QVector3D> &vec)
{
    if(vec.size() == 0)
        return -1;

    double miny = MAX_INT+1000000;
    int id = 0;
    for(unsigned int i=0; i<vec.size(); i++)
    {
        if(miny > vec[i].y())
        {
            miny = vec[i].y();
            id = i;

            this->m_rootPosition = vec[i];
        }
    }

    return id;
}

void Tree::displayFinalSkeletonWithColors(bool b)
{
    this->m_isSkeletonBeColored = b;

    if(this->m_process_stage != _STAGE_7 || this->m_binRoot == NULL)
        return;

    this->createGraphVBOFromBin();
}

// [1] 从文件中加载三维模型的数据（顺便进行数据压缩，删除掉重复的点）
//void Tree::readDataFromFile(QString filename)
//{
//    qDebug()<<"[1] 开始加载点云数据:"<<filename;
//    initializeOpenGLFunctions();
//    enterCurrentStageMode(_STAGE_1);
//    // 包围盒信息
//    double minx = 9999999;
//    double maxx = -999999;
//    double miny = 9999999;
//    double maxy = -999999;
//    double minz = 9999999;
//    double maxz = -999999;

//    // 点云数量
//    int prevCount = 0;
//    int lastCount = 0;

//    QFile file(filename);
//    if(!file.open(QIODevice::ReadOnly))
//        return;
//    QTextStream ts(&file);


//    while(!ts.atEnd())
//    {
//        QStringList line = ts.readLine().split(" ");
//        line.removeAll("");

//        if(line.size() >= 4 && line[0] == "v")
//        {
//            QVector3D vec(line[1].toDouble(),line[2].toDouble(),line[3].toDouble());

//            m_vertexs.push_back(vec);

//            minx=minx > vec.x()?vec.x():minx;
//            maxx=maxx < vec.x()?vec.x():maxx;
//            miny=miny > vec.y()?vec.y():miny;
//            maxy=maxy < vec.y()?vec.y():maxy;
//            minz=minz > vec.z()?vec.z():minz;
//            maxz=maxz < vec.z()?vec.z():maxz;
//        }
//    }
//    file.close();

//    prevCount = this->m_vertexs.size();
//    qDebug()<<"  - 压缩前点云数量:"<<prevCount;

//    // 修正到不会超过100的范围
//    double radio,dx,dy,dz;
//    dx = maxx - minx;
//    dy = maxy - miny;
//    dz = maxz - minz;

//    double max = ((dx>dy)?dx:dy)>dz?((dx>dy)?dx:dy):dz;
//    radio = 100.0/max;
//    for(unsigned int i=0; i<m_vertexs.size(); i++)
//    {
//        m_vertexs[i].setX(m_vertexs[i].x());
//        m_vertexs[i].setY(m_vertexs[i].y());
//        m_vertexs[i].setZ(m_vertexs[i].z());

//        m_vertexs[i].setX(m_vertexs[i].x()*radio);
//        m_vertexs[i].setY(m_vertexs[i].y()*radio);
//        m_vertexs[i].setZ(m_vertexs[i].z()*radio);
//    }

//    // 压缩点云数据，以提出重复点
//    QList<int> removedIndex;    // 待删除点的索引

//    PointCloud<double> cloud;   // 建立PointCloud
//    VectorToPointCloud(m_vertexs,cloud);

//    KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(30) );
//    index.buildIndex();

//    for(unsigned int i=0; i<this->m_vertexs.size(); i++)
//    {
//        if(removedIndex.contains(i))
//            continue;
//        std::vector< std::pair<size_t,double> > ret_matches;
//        nanoflann::SearchParams params;

//        // 待查询点
//        const double query_pt[3] = {this->m_vertexs[i].x(),this->m_vertexs[i].y(),this->m_vertexs[i].z()};

//        // 范围查询
//        const int nMatches = index.radiusSearch(&query_pt[0], 0.1, ret_matches, params);

//        for(int k=0; k<nMatches; k++)
//        {
//            if(fabs(ret_matches[k].second)<=0.0000001 && i!=ret_matches[k].first)  // 距离为0 并且不是本身
//                removedIndex.push_back(ret_matches[k].first);
//        }
//    }

//    cloud.pts.clear();

//    qSort(removedIndex.begin(),removedIndex.end(),qGreater<int>());

//    for(unsigned int i=0; i<removedIndex.size(); i++)
//    {
//        if(i==0)
//            this->m_vertexs.removeAt(removedIndex[i]);
//        else
//        {
//            if(removedIndex[i-1] != removedIndex[i])
//            {
//                this->m_vertexs.removeAt(removedIndex[i]);
//            }
//        }
//    }
//    lastCount = this->m_vertexs.size();
//    qDebug()<<"  - 压缩后点云数量:"<<lastCount<<QString("(压缩率:%1\%)").arg(100*(float)lastCount/(float)prevCount);

//    savePointsTo(this->m_vertexs,PATH_ORIGIN_POINTS);

//    this->m_centerPos = getCenterFrom(this->m_vertexs);

//    this->createPointVBO(this->m_vertexs);
//    this->createGraphVBO();
//}


void Tree::readDataFromFile(QString filename)
{
    qDebug()<<"[1] 开始加载点云数据:"<<filename;
    initializeOpenGLFunctions();
    enterCurrentStageMode(_STAGE_1);
    // 包围盒信息
    double minx = 9999999;
    double maxx = -999999;
    double miny = 9999999;
    double maxy = -999999;
    double minz = 9999999;
    double maxz = -999999;

    // 点云数量
    int prevCount = 0;
    int lastCount = 0;

    QFile file(filename);
    if(!file.open(QIODevice::ReadOnly))
        return;
    QTextStream ts(&file);


    while(!ts.atEnd())
    {
        QStringList line = ts.readLine().split(" ");
        line.removeAll("");

        if(line.size() >= 3)
        {

            QVector3D vec(line[0].toDouble(),line[1].toDouble(),line[2].toDouble());

            m_vertexs.push_back(vec);

            minx=minx > vec.x()?vec.x():minx;
            maxx=maxx < vec.x()?vec.x():maxx;
            miny=miny > vec.y()?vec.y():miny;
            maxy=maxy < vec.y()?vec.y():maxy;
            minz=minz > vec.z()?vec.z():minz;
            maxz=maxz < vec.z()?vec.z():maxz;
        }
    }
    file.close();




    prevCount = this->m_vertexs.size();
    qDebug()<<"  - 压缩前点云数量:"<<prevCount;

    // 修正到不会超过100的范围
    double radio,dx,dy,dz;
    dx = maxx - minx;
    dy = maxy - miny;
    dz = maxz - minz;

    double max = ((dx>dy)?dx:dy)>dz?((dx>dy)?dx:dy):dz;
    radio = 100.0/max;
    for(unsigned int i=0; i<m_vertexs.size(); i++)
    {
        m_vertexs[i].setX(m_vertexs[i].x());
        m_vertexs[i].setY(m_vertexs[i].y());
        m_vertexs[i].setZ(m_vertexs[i].z());

        m_vertexs[i].setX(m_vertexs[i].x()*radio);
        m_vertexs[i].setY(m_vertexs[i].y()*radio);
        m_vertexs[i].setZ(m_vertexs[i].z()*radio);
    }

    // 压缩点云数据，以提出重复点
    QList<int> removedIndex;    // 待删除点的索引

    PointCloud<double> cloud;   // 建立PointCloud
    VectorToPointCloud(m_vertexs,cloud);

    KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(30) );
    index.buildIndex();

    for(unsigned int i=0; i<this->m_vertexs.size(); i++)
    {
        if(removedIndex.contains(i))
            continue;
        std::vector< std::pair<size_t,double> > ret_matches;
        nanoflann::SearchParams params;

        // 待查询点
        const double query_pt[3] = {this->m_vertexs[i].x(),this->m_vertexs[i].y(),this->m_vertexs[i].z()};

        // 范围查询
        const int nMatches = index.radiusSearch(&query_pt[0], 0.1, ret_matches, params);

        for(int k=0; k<nMatches; k++)
        {
            if(fabs(ret_matches[k].second)<=0.0000001 && i!=ret_matches[k].first)  // 距离为0 并且不是本身
                removedIndex.push_back(ret_matches[k].first);
        }
    }

    cloud.pts.clear();

    qSort(removedIndex.begin(),removedIndex.end(),qGreater<int>());

    for(unsigned int i=0; i<removedIndex.size(); i++)
    {
        if(i==0)
            this->m_vertexs.removeAt(removedIndex[i]);
        else
        {
            if(removedIndex[i-1] != removedIndex[i])
            {
                this->m_vertexs.removeAt(removedIndex[i]);
            }
        }
    }
    lastCount = this->m_vertexs.size();
    qDebug()<<"  - 压缩后点云数量:"<<lastCount<<QString("(压缩率:%1\%)").arg(100*(float)lastCount/(float)prevCount);

    savePointsTo(this->m_vertexs,PATH_ORIGIN_POINTS);

    this->m_centerPos = getCenterFrom(this->m_vertexs);

    this->createPointVBO(this->m_vertexs);
    this->createGraphVBO();
}

void Tree::voxelization(QVector<QVector3D> &vec)
{

}

// [1] 从输入的原始点云中获取树干区域的点云
void Tree::getBranchAreaFromPoints(double searchRadius)
{
    qDebug()<<"[2] 开始提取树干区域点云:"<<QString("(r=%1)").arg(searchRadius);

    this->enterCurrentStageMode(_STAGE_2);

    // 首先，从文件中读取回来原始点云
    loadPointsFrom(this->m_vertexs,PATH_ORIGIN_POINTS);
    qDebug()<<"  - 原始点数量:"<<this->m_vertexs.size();

    if(!this->m_vertexs.size())
        return;

    PointCloud<double> cloud;
    VectorToPointCloud(this->m_vertexs,cloud);

    KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(30) );
    index.buildIndex();

    QVector<int>  rootPart;   // 当前正在查找的区域
    QVector<bool> visited;    // 已经被访问过了么

    int m_rootIndex = locateRootindex(this->m_vertexs);

    rootPart.push_back(m_rootIndex);

    visited.resize(m_vertexs.size());
    for(int i=0; i<visited.size(); i++)
        visited[i] = false;
    visited[m_rootIndex] = true;

    for(int i=0; i<rootPart.size(); i++)
    {
        std::vector< std::pair<size_t,double> > ret_matches;

        nanoflann::SearchParams params;

        // 待查询点
        const double query_pt[3] = {m_vertexs[rootPart[i]].x(),m_vertexs[rootPart[i]].y(),m_vertexs[rootPart[i]].z()};

        // 范围查询
        const int nMatches = index.radiusSearch(&query_pt[0], searchRadius, ret_matches, params);

        for(int k=0; k<nMatches; k++)
        {
            if(visited[ret_matches[k].first] == true || i==ret_matches[k].first)
                continue;

            rootPart.push_back(ret_matches[k].first);

            visited[ret_matches[k].first] = true;
        }
    }

    for(int i=0; i<rootPart.size(); i++)
    {
        m_curVertexes.push_back(m_vertexs[rootPart[i]]);
    }

    for(int i=0; i<m_vertexs.size(); i++)
    {
        if(visited[i] == false)
            m_leafPart.push_back(m_vertexs[i]);
    }

    qDebug()<<"  - 树干区域点云数量:"<<this->m_curVertexes.size()<<QString("(%1\%)")
              .arg(100*(float)m_curVertexes.size()/(float)m_vertexs.size());


    // 体素化
//    BBox bbox;
//    bbox.create(3.0);
//    bbox.addPoints(this->m_curVertexes);
//    bbox.getVertexArray(this->m_curVertexes);
//    bbox.clear();

    // 文件处理 收尾工作：
    this->m_vertexs.clear();

    savePointsTo(this->m_curVertexes,PATH_BRANCH_PART);
    savePointsTo(this->m_leafPart,PATH_LEAVES_PART);
    this->m_leafPart.clear();
}

// 获取连通图
void Tree::generateConnectedGraph(double searchRadius)
{
    qDebug()<<"[3] 开始建立连通图:"<<QString("(r=%1)").arg(searchRadius);

    enterCurrentStageMode(_STAGE_3);

    loadPointsFrom(this->m_curVertexes,PATH_BRANCH_PART);

    if(this->m_curVertexes.size() == 0 && this->m_vertexs.size()!=0)
    {
        this->m_curVertexes = this->m_vertexs;
        savePointsTo(this->m_curVertexes,PATH_BRANCH_PART);
    }

    if(this->m_curVertexes.size()==0)
        return;

    this->m_branchGraph.clear();
    this->m_branchGraph.resize(this->m_curVertexes.size());


    // 创建KD树
    PointCloud<double> cloud;
    cloud.pts.resize(m_curVertexes.size());
    for(int i=0; i<m_curVertexes.size(); i++)
    {
        cloud.pts[i].x = m_curVertexes[i].x();
        cloud.pts[i].y = m_curVertexes[i].y();
        cloud.pts[i].z = m_curVertexes[i].z();
    }
    typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double> > ,PointCloud<double>,3 /* dim */>
            my_kd_tree_t;

    my_kd_tree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(30 /* max leaf */) );
    index.buildIndex();

    int count = 0;
    for(unsigned int i=0; i<this->m_curVertexes.size(); i++)
    {
        std::vector< std::pair<size_t,double> > ret_matches;
        nanoflann::SearchParams params;

        // 待查询点
        const double query_pt[3] = {this->m_curVertexes[i].x(),this->m_curVertexes[i].y(),this->m_curVertexes[i].z()};

        // 范围查询
        const int nMatches = index.radiusSearch(&query_pt[0], searchRadius, ret_matches, params);

        for(int k=0; k<nMatches; k++)
        {
            if(i==ret_matches[k].first)
                continue;

            m_branchGraph[i].push_back(QPair<int,double>(ret_matches[k].first,ret_matches[k].second));
            count++;
        }
    }

    // 至此联通树已经建立完成
    qDebug()<<"  - 连通图边数:"<<count;
    createPointVBO(this->m_curVertexes);
}

void Tree::dijkstraMinGraph()
{
    qDebug()<<"[4] 开始建立最小生成树";

    if(this->m_process_stage != _STAGE_3)
        return;

    loadPointsFrom(this->m_curVertexes,PATH_BRANCH_PART);
    int rootIndex = locateRootindex(this->m_curVertexes);

    if(rootIndex == -1)
        return;

    if(this->m_branchGraph.size()==0)
        return;

    enterCurrentStageMode(_STAGE_4);

    clearMinTree();

    // 初始化访问数组
    QVector<bool> isVisted;   // 000001000000
    isVisted.fill(false,this->m_curVertexes.size());
    isVisted[rootIndex] = true;

    // 初始化dist
    QVector<double> dist;
    QVector<int>    prev;
    dist.fill(MAX_INT,this->m_curVertexes.size());
    prev.fill(-1,this->m_curVertexes.size());

    dist[rootIndex] = 0;
    for(unsigned int i=0;i<this->m_branchGraph[rootIndex].size(); i++)
    {
        dist[this->m_branchGraph[rootIndex][i].first]
                = this->m_branchGraph[rootIndex][i].second;
    }


    this->m_curVertexes.clear();
    this->m_leafPart.clear();
    this->createGraphVBO();
}

void Tree::divideIntoBins1(double r)
{
    qDebug()<<"[5] 开始划分Bins，将所有的点根据距离进行划分";

    // 遍历树 计算level
    if(!this->m_rootMinGraph)
        return;

    enterCurrentStageMode(_STAGE_5);

    // 清空原有数据
    for(unsigned int i=0; i<bins.size(); i++)
    {
        bins[i].childs.clear();
        bins[i].pts.clear();
        bins[i].ptsId.clear();
    }
    bins.clear();

    m_maxBinLevel = -1;
    QVector<TreeNode*> branchs;
    branchs.push_back(this->m_rootMinGraph);
    m_branchGraph.clear();

    while(branchs.size()!=0)
    {
        TreeNode* cur = branchs.front();
        branchs.pop_front();

        cur->level = cur->dist/r;  // 计算每个点的level

        if(cur->level > m_maxBinLevel)  // 记录一下最大的level层次
            m_maxBinLevel = cur->level;

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            branchs.push_back(cur->childs[i]);
        }
    }

    // 遍历树  将点划分为bins
    branchs.clear();
    branchs.push_back(this->m_rootMinGraph);

    bins.resize(m_maxBinLevel+1);

    while(branchs.size()!=0)
    {
        TreeNode* cur = branchs.front();
        branchs.pop_front();

        bins[cur->level].pts.append(cur->pos);

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            branchs.push_back(cur->childs[i]);
        }
    }

    createPointVBOFromBins(bins);
    qDebug()<<"  - 初步的bins数目: "<<bins.size();
}

void Tree::divideIntoBins2(double minGapScale)
{
    // 这里bins已经初步划分好了
    // 接着，将bins中距离比较远的也分开
    qDebug()<<"[5] 对Bin进行划分";

    if(bins.size() == 0)
        return;
    enterCurrentStageMode(_STAGE_6);

    clearBinTree();
    createPointVBOFromBins(finalbins);

}

void Tree::getFinalSkeleton()
{
    if(!this->m_binRoot)
        return;

    enterCurrentStageMode(_STAGE_7);

    // 首先构造深度
    QVector<Bin*> stacks;

    stacks.push_back(this->m_binRoot);
    this->m_binRoot->parentBin = NULL;

    QVector<Bin*> bufferBin;
    while(stacks.size()!=0)
    {
        Bin* cur = stacks.front();
        stacks.pop_front();
        cur->depth = 0;

        if(cur->childs.size() == 0)
            bufferBin.push_back(cur);

        for(unsigned int i=0; i<cur->childs.size();i++)
        {
            cur->childs[i]->parentBin = cur;
            stacks.push_back( cur->childs[i] );
        }
    }

    int curDepth = 1;
    while(true)
    {
        if(bufferBin.size()==0)
            break;
        for(unsigned int i=0; i<bufferBin.size(); i++)
        {
            if(bufferBin[i]->depth < curDepth)  // 如果有更深的depth则进行更新
                bufferBin[i]->depth = curDepth;
        }

        QVector<Bin*> nextBufferBin;
        for(unsigned int i=0; i<bufferBin.size(); i++)
        {
            if(bufferBin[i]->parentBin!=NULL)
                nextBufferBin.push_back(bufferBin[i]->parentBin);
        }

        bufferBin = nextBufferBin;
        curDepth++;
    }
    stacks.push_back(this->m_binRoot);

    // 创建骨架
    QVector<QVector3D> nodes;
    QVector< QPair<QVector3D,QVector3D> > graph;
    QVector< QPair<int,int> > graphDepth;

    while(stacks.size()!=0)
    {
        Bin* cur = stacks.front();
        stacks.pop_front();

        nodes.push_back(cur->centerPos);

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            graph.push_back(QPair<QVector3D,QVector3D>(cur->centerPos,cur->childs[i]->centerPos));
            graphDepth.push_back(QPair<int,int>(cur->depth,cur->childs[i]->depth));
            stacks.push_back(cur->childs[i]);
        }
    }

    createPointVBO(nodes);
    this->m_skeletonPts = nodes;

    createGraphVBOFrom(graph,graphDepth);
}

void Tree::optimizeSkeleton()
{
    if(this->m_process_stage != _STAGE_7 || this->m_binRoot == NULL)
        return;

    // 已经有了深度
    QVector<Bin*> stack;
    stack.push_back(this->m_binRoot);
    QVector<QVector3D> nodes;
    QVector< QPair<QVector3D,QVector3D> > graph;
    QVector< QPair<int,int> > graphDepth;
    while(stack.size()!=0)
    {
        Bin* cur = stack.front();
        stack.pop_front();

        nodes.push_back(cur->centerPos);

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            if(cur->childs[i]->depth <= 1 && (cur->depth-cur->childs[i]->depth)>3)
            {
                cur->childs.removeAt(i);
                i--;
                continue;
            }
            stack.push_back(cur->childs[i]);
            graphDepth.push_back(QPair<int,int>(cur->depth,cur->childs[i]->depth));
            graph.push_back(QPair<QVector3D,QVector3D>(cur->centerPos,cur->childs[i]->centerPos));
        }
    }

    //
    createPointVBO(nodes);
    this->m_skeletonPts = nodes;

    createGraphVBOFrom(graph,graphDepth);
}

void Tree::createGraphVBOFromBin()
{
    if(this->m_process_stage != _STAGE_7 || this->m_binRoot == NULL)
        return;

    // 已经有了深度
    QVector<Bin*> stack;
    stack.push_back(this->m_binRoot);
    QVector<QVector3D> nodes;
    QVector< QPair<QVector3D,QVector3D> > graph;
    QVector< QPair<int,int> > graphDepth;
    while(stack.size()!=0)
    {
        Bin* cur = stack.front();
        stack.pop_front();

        nodes.push_back(cur->centerPos);

        for(unsigned int i=0; i<cur->childs.size(); i++)
        {
            stack.push_back(cur->childs[i]);
            graphDepth.push_back(QPair<int,int>(cur->depth,cur->childs[i]->depth));
            graph.push_back(QPair<QVector3D,QVector3D>(cur->centerPos,cur->childs[i]->centerPos));
        }
    }

    //
    createPointVBO(nodes);
    this->m_skeletonPts = nodes;

    createGraphVBOFrom(graph,graphDepth);
}

void Tree::createGraphVBOFrom(const QVector<QPair<QVector3D, QVector3D> > &graph, const QVector<QPair<int, int> > &graphDepth)
{
    // 创建图
    int maxDepth=0;  // 找到深度的最大值
    if(this->m_isSkeletonBeColored)
    {
        for(unsigned int i=0; i<graphDepth.size(); i++)
        {
            if(maxDepth < graphDepth[i].first)
                maxDepth = graphDepth[i].first;
            if(maxDepth < graphDepth[i].second)
                maxDepth = graphDepth[i].second;
        }
    }
    if(this->m_graphVBO.isCreated())
        this->m_graphVBO.release();

    QVector<GLfloat> data;

    for(unsigned int i=0; i<graph.size(); i++)
    {
        data.push_back(graph[i].first.x());
        data.push_back(graph[i].first.y());
        data.push_back(graph[i].first.z());
        if(this->m_isSkeletonBeColored)
        {
            data.push_back(((float)maxDepth-graphDepth[i].first)/(float)maxDepth);
            data.push_back( 2*(0.5-fabs(((graphDepth[i].first)/(float)maxDepth)-0.5) ) ); // [-0.5,0,0.5]
            data.push_back(graphDepth[i].first/(float)maxDepth);
        }
        else
        {
            data.push_back(0.0);
            data.push_back(0.6);
            data.push_back(0.0);
        }
        data.push_back(graph[i].second.x());
        data.push_back(graph[i].second.y());
        data.push_back(graph[i].second.z());

        if(this->m_isSkeletonBeColored)
        {
            data.push_back(((float)maxDepth-graphDepth[i].second)/(float)maxDepth);
            data.push_back( 2*(0.5-fabs(((graphDepth[i].second)/(float)maxDepth)-0.5)) ); // [-0.5,0,0.5]
            data.push_back(graphDepth[i].second/(float)maxDepth);
        }
        else
        {
            data.push_back(0.0);
            data.push_back(0.6);
            data.push_back(0.0);
        }
    }

    this->m_graphVBO.create();
    this->m_graphVBO.bind();
    this->m_graphVBO.allocate(data.constData(),data.count()*sizeof(GLfloat));

    this->m_graphVBO_Vertex_Count = data.size()/6.0;
}

void Tree::createPointVBO(const QVector<QVector3D> &vecs, QVector<QVector3D> colors)
{
    if(this->m_pointVBO.isCreated())
        this->m_pointVBO.release();

    QVector<GLfloat> data;

    if(colors.size() == 0)
        colors.fill(QVector3D(0.0,0.0,0.0),vecs.size());

    for(unsigned int i=0; i<vecs.size(); i++)
    {
        data.push_back(vecs[i].x());
        data.push_back(vecs[i].y());
        data.push_back(vecs[i].z());

        data.push_back(colors[i].x());
        data.push_back(colors[i].y());
        data.push_back(colors[i].z());
    }
    colors.clear();

    this->m_pointVBO.create();
    this->m_pointVBO.bind();
    this->m_pointVBO.allocate(data.constData(),data.count()*sizeof(GLfloat));

    this->m_pointVBO_Vertex_Count = data.size()/6.0;
}

void Tree::createPointVBOFromBins(QVector<Bin*>& bins)
{
    if(this->m_pointVBO.isCreated())
        this->m_pointVBO.release();

    qsrand(QTime::currentTime().msec());
    QVector<GLfloat> data;

    QVector<QVector3D> colors;
//    colors.resize(bins.size());

//    for(unsigned int i=0; i<bins.size(); i++)
//    {
//        colors.push_back( QVector3D((qrand()%255),
//                                    (qrand()%255),(qrand()%255)));
//        qDebug()<<colors[i];
//    }

    colors<<QVector3D(50,162,240)<<QVector3D(251,148,3)<<QVector3D(31,114,70)
         <<QVector3D(41,85,152)<<QVector3D(248,101,106)<<QVector3D(22,224,109)
        <<QVector3D(211,226,99)<<QVector3D(248,101,206)<<QVector3D(31,14,70);

    for(unsigned int i=0; i<colors.size(); i++)
    {
        colors[i] /= 255.0;
    }

    for(unsigned int i=0; i<bins.size(); i++)
    {
        for(unsigned int k=0; k<bins[i]->pts.size(); k++)
        {
            data.push_back(bins[i]->pts[k].x());
            data.push_back(bins[i]->pts[k].y());
            data.push_back(bins[i]->pts[k].z());

            data.push_back(colors[i%colors.size()].x());
            data.push_back(colors[i%colors.size()].y());
            data.push_back(colors[i%colors.size()].z());
        }
    }
    colors.clear();

    this->m_pointVBO.create();
    this->m_pointVBO.bind();
    this->m_pointVBO.allocate(data.constData(),data.count()*sizeof(GLfloat));

    this->m_pointVBO_Vertex_Count = data.size()/6.0;
}

void Tree::createPointVBOFromBins(QVector<Bin>& bins)
{
    if(this->m_pointVBO.isCreated())
        this->m_pointVBO.release();

    qsrand(QTime::currentTime().msec());
    QVector<GLfloat> data;

    QVector<QVector3D> colors;
//    colors.resize(bins.size());

//    for(unsigned int i=0; i<bins.size(); i++)
//    {
//        colors.push_back( QVector3D((qrand()%255),
//                                    (qrand()%255),(qrand()%255)));
//        qDebug()<<colors[i];
//    }

    colors<<QVector3D(50,162,240)<<QVector3D(251,148,3)<<QVector3D(31,114,70)
         <<QVector3D(41,85,152)<<QVector3D(248,101,106)<<QVector3D(22,224,109)
        <<QVector3D(211,226,99)<<QVector3D(248,101,206)<<QVector3D(31,14,70);

    for(unsigned int i=0; i<colors.size(); i++)
    {
        colors[i] /= 255.0;
    }

    for(unsigned int i=0; i<bins.size(); i++)
    {
        for(unsigned int k=0; k<bins[i].pts.size(); k++)
        {
            data.push_back(bins[i].pts[k].x());
            data.push_back(bins[i].pts[k].y());
            data.push_back(bins[i].pts[k].z());

            data.push_back(colors[i%colors.size()].x());
            data.push_back(colors[i%colors.size()].y());
            data.push_back(colors[i%colors.size()].z());
        }
    }
    colors.clear();

    this->m_pointVBO.create();
    this->m_pointVBO.bind();
    this->m_pointVBO.allocate(data.constData(),data.count()*sizeof(GLfloat));

    this->m_pointVBO_Vertex_Count = data.size()/6.0;
}


void Tree::createGraphVBO()
{
    if(this->m_graphVBO.isCreated())
        this->m_graphVBO.release();

    QVector<GLfloat> data;

    for(unsigned int i=0; i<this->m_branchGraph.size(); i++)
    {
        for(unsigned int k=0; k<this->m_branchGraph[i].size(); k++)
        {
            data.push_back(this->m_curVertexes[i].x());
            data.push_back(this->m_curVertexes[i].y());
            data.push_back(this->m_curVertexes[i].z());

            data.push_back(this->m_curVertexes[this->m_branchGraph[i][k].first].x());
            data.push_back(this->m_curVertexes[this->m_branchGraph[i][k].first].y());
            data.push_back(this->m_curVertexes[this->m_branchGraph[i][k].first].z());
        }
    }

    this->m_graphVBO.create();
    this->m_graphVBO.bind();
    this->m_graphVBO.allocate(data.constData(),data.count()*sizeof(GLfloat));

    this->m_graphVBO_Vertex_Count = data.size()/3.0;

    //qDebug()<<"m_graphVBO_Vertex_Count"<<m_graphVBO_Vertex_Count;
}

void Tree::drawPointVBO(QOpenGLShaderProgram *&program, const QMatrix4x4 &modelMat)
{
#define VERTEX_ATTRIBUTE_BRANCH 0
#define COLOR_ATTRIBUTE_BRANCH 1

    program->setUniformValue("mat_model",modelMat);

    if(this->m_pointVBO_Vertex_Count <= 0)
        return;

    this->m_pointVBO.bind();
    program->enableAttributeArray(VERTEX_ATTRIBUTE_BRANCH);
    program->enableAttributeArray(COLOR_ATTRIBUTE_BRANCH);
    program->setAttributeBuffer(VERTEX_ATTRIBUTE_BRANCH,  GL_FLOAT, 0,                3,6*sizeof(GLfloat));
    program->setAttributeBuffer(COLOR_ATTRIBUTE_BRANCH,  GL_FLOAT, 3*sizeof(GLfloat),3,6*sizeof(GLfloat));

    glDrawArrays(GL_POINTS,0,this->m_pointVBO_Vertex_Count);

    program->disableAttributeArray(COLOR_ATTRIBUTE_BRANCH);
}

void Tree::drawGraphVBOWithOutColors(QOpenGLShaderProgram *&program, const QMatrix4x4 &modelMat)
{
#define VERTEX_ATTRIBUTE_BRANCH 0
#define COLOR_ATTRIBUTE_BRANCH 1

    if(this->m_process_stage == _STAGE_7)
    {
        program->setUniformValue("mat_model",modelMat);

        if(this->m_graphVBO_Vertex_Count <= 0)
            return;

        this->m_graphVBO.bind();
        program->enableAttributeArray(VERTEX_ATTRIBUTE_BRANCH);
        program->enableAttributeArray(COLOR_ATTRIBUTE_BRANCH);
        program->setAttributeBuffer(VERTEX_ATTRIBUTE_BRANCH,  GL_FLOAT, 0,                3,6*sizeof(GLfloat));
        program->setAttributeBuffer(COLOR_ATTRIBUTE_BRANCH,  GL_FLOAT, 3*sizeof(GLfloat),3,6*sizeof(GLfloat));

        glDrawArrays(GL_LINES,0,this->m_graphVBO_Vertex_Count);
    }
    else
    {
        program->setUniformValue("mat_model",modelMat);

        if(this->m_graphVBO_Vertex_Count <= 0)
            return;

        this->m_graphVBO.bind();
        program->enableAttributeArray(VERTEX_ATTRIBUTE_BRANCH);
        //program->enableAttributeArray(COLOR_ATTRIBUTE_BRANCH);
        program->setAttributeBuffer(VERTEX_ATTRIBUTE_BRANCH,  GL_FLOAT, 0,                3,3*sizeof(GLfloat));
        const GLfloat color[3] = {0.0,0.6,0.0};
        program->setAttributeValue("a_color",QVector3D(0.0,0.6,0.0));

        glDrawArrays(GL_LINES,0,this->m_graphVBO_Vertex_Count);
    }
}
