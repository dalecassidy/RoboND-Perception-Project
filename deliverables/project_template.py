#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    
    # TODO: Convert ROS msg to PCL data

    pcl_data = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering

    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(3)
    x = 1.0
    outlier_filter.set_std_dev_mul_thresh(x)
    pcl_data = outlier_filter.filter()

    #pcl_filtered_pub.publish(pcl_to_ros(pcl_data))

    # TODO: Voxel Grid Downsampling

    vox = pcl_data.make_voxel_grid_filter()
    LEAF_SIZE = 0.01

    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()
    filename = 'voxel_downsampled.pcd'
    # pcl.save(cloud_filtered, filename)

    # TODO: PassThrough Filter

    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'

    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1

    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(-.5,.4)
    cloud_filtered = passthrough.filter()

    filename = 'pass_through_filtered.pcd'
    pcl.save(cloud_filtered, filename)
    pcl_passthrough_pub.publish(pcl_to_ros(cloud_filtered))
    # TODO: RANSAC Plane Segmentation

    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = .01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers

    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    filename = 'extracted_inliers.pcd'
    #pcl.save(extracted_inliers, filename)

    # objects on table
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    filename = 'extracted_outliers.pcd'
    #pcl.save(extracted_outliers, filename)

    cloud_table = extracted_inliers
    cloud_objects = extracted_outliers

    # TODO: Euclidean Clustering

    white_cloud  = XYZRGB_to_XYZ(cloud_objects)

    #pcl.save(white_cloud, 'white_cloud.pcd')

    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.06)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(2000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract() 

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages

    ros_cluster_cloud = pcl_to_ros(cluster_cloud)    
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)

    # TODO: Publish ROS messages

    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud) 

    #check pcl data
    pcl_check_pub.publish(pcl_msg)

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
      
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
        

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects

    detected_objects_pub.publish(detected_objects)

    # Calculate centroids
    object_dict = dict()
  
    for object in detected_objects:
        label = object.label 
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroid=(np.mean(points_arr, axis=0)[:3])
        object_dict[label] = centroid

    # Open pick list

    dropbox_list = rospy.get_param('/dropbox')
    dropbox_left_pos = dropbox_list[0]['position']
    dropbox_right_pos = dropbox_list[1]['position']
    print(dropbox_right_pos)

    object_list_param = rospy.get_param('/object_list')

    test_scene_num = Int32()
    test_scene_num.data = 3 

    object_name = String()
    arm_name = String()

    pick_pose = Pose()
    place_pose = Pose()

    # Loop through pick list 
    yaml_dict_list = []
    for i in range(len(object_list_param)):       
        object_name.data = object_list_param[i]['name']
        if object_name.data in object_dict.keys():
	     object_group = object_list_param[i]['group']        
	     if object_group == 'red':
                  arm_name.data = 'left'
		  place_pose.position.x = dropbox_left_pos[0]
                  place_pose.position.y = dropbox_left_pos[1]
                  place_pose.position.z = dropbox_left_pos[2]
	     else:
	          arm_name.data = 'right'
                  place_pose.position.x = dropbox_right_pos[0]
                  place_pose.position.y = dropbox_right_pos[1]
                  place_pose.position.z = dropbox_right_pos[2]
	     pick_pose.position.x = np.asscalar(object_dict[object_name.data][0])
             pick_pose.position.y = np.asscalar(object_dict[object_name.data][1])
             pick_pose.position.z = np.asscalar(object_dict[object_name.data][2])             
	 
	     yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
	     yaml_dict_list.append(yaml_dict)

    print(yaml_dict_list)
    yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"
    send_to_yaml(yaml_filename, yaml_dict_list)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    #try:
    #    pr2_mover(detected_objects)
    #except rospy.ROSInterruptException:
    #    pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # TODO: ROS node initialization

    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers

    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers

    pcl_check_pub = rospy.Publisher("/pcl_check", PointCloud2, queue_size=1)
    pcl_filtered_pub = rospy.Publisher("/pcl_filter", PointCloud2, queue_size=1)
    pcl_passthrough_pub = rospy.Publisher("pcl_passthrough", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    # TODO: Load Model From disk

    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()




