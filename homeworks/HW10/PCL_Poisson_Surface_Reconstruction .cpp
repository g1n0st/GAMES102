#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>

#include <fstream>

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);  
    if (pcl::io::loadPCDFile("../bunny.pcd" , *cloud) == -1) { 
        PCL_ERROR("Read pcd file failed!\n");  
        return -1;  
    }  

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);  
    pcl::NormalEstimation<pcl::PointXYZ , pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);  

    tree->setInputCloud(cloud);  

    n.setInputCloud(cloud); 
	n.setSearchMethod(tree);  
	n.setKSearch(20);  
	n.compute(*normals);
	
	pcl::concatenateFields(*cloud , *normals , *cloud_with_normals);  
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);  
	
	tree2->setInputCloud(cloud_with_normals);  
	
	pcl::Poisson<pcl::PointNormal> pn;
	pn.setSearchMethod(tree2); 
	pn.setInputCloud(cloud_with_normals); 
	pn.setConfidence(false);	
    pn.setManifold(false);
	pn.setOutputPolygons(false);	
    pn.setIsoDivide(8);  
	pn.setSamplesPerNode(3);	

    pcl::PolygonMesh mesh; 
	pn.performReconstruction(mesh); 
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));

	viewer->setBackgroundColor(0 , 0 , 0); 
	viewer->addPolygonMesh(mesh , "mesh"); 
	viewer->initCameraParameters();  
	
	while (!viewer->wasStopped()) { 
	    viewer->spinOnce(100);
	}
	
	return 0;
}
