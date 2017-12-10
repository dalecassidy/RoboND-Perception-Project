## Project: Perception Pick & Place
[//]: # (Image References)

[image1]: ./images/scene1.jpg
[image2]: ./images/scene2.jpg
[image3]: ./images/scene3.jpg
[image4]: ./images/extracted_outliers.jpg
[image5]: ./images/pass_through_filtered.jpg
[image6]: ./images/segmentation.jpg
[image7]: ./images/confusion.jpg
[image8]: ./images/nconfusion.jpg

For this project, I implemented a perception pipeline. First I used pass through and RANSAC filtering
on a point cloud to isolate the objects off of a table. I then applied Euclidean clustering 
to separate individual items and then I performed object recognition using an SVM trained on 8 different items.
Having assigned the objects' labels, I calculated their centroids. I then outputted the data to yaml files 
which describe the objects in their repsective test scenes. 

The SVM was trained on all 8 objects found in pick_list_3.yaml. This was done with a camera in Gazebo
taking several pictures of each object in several different orientations and then extracting color and normal
features from those images. 

The following is the code I wrote for the 3 major sections of this project: filtering, clustering and recognition. I describe my approach in
the comments of the code and add supporting images along the way. At the end I discuss
what I might have tried if I had more time to work on the project.

4 files have been submitted on GitHub as part of this project: project_template.py, output_1.yaml, output_2.yaml and output_3.yaml. They are under
the deliverables folder.

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

The first part of the code removes noisy data from the image, then downsamples it before
isolating the objects from the table.

```
def pcl_callback(pcl_msg):
   
    # TODO: Convert ROS msg to PCL data

    pcl_data = ros_to_pcl(pcl_msg)
   
    # TODO: Statistical Outlier Filtering
    # Removes noisy data from the image.

    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(3)
    x = 1.0
    outlier_filter.set_std_dev_mul_thresh(x)
    pcl_data = outlier_filter.filter()

    # publish cloud data to ROS node for test

    pcl_filtered_pub.publish(pcl_to_ros(pcl_data))

    # TODO: Voxel Grid Downsampling
    # Done because there is too much unnecessary data if you use every pixel

    vox = pcl_data.make_voxel_grid_filter()
    LEAF_SIZE = 0.01

    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()
    filename = 'voxel_downsampled.pcd'
    # pcl.save(cloud_filtered, filename)

    # TODO: PassThrough Filter
    # This section only shows the relevant section of the table. To do this
    # I had to remove data from both the z and y axes

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
    # This section is used to extract only the objects from the scene 
    # removing the table

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
```

The following is what the point cloud looks like after the pass through filter.

![alt text][image5]

This image shows what the point cloud looks like after RANSAC is run on the previous image,
separating the table from the objects.

![alt text][image4]


#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

This section clusters and segments the objects.

```
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
```

You can see from the following image that the objects have been clustered and segmented.

![alt text][image6]

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

The following is the histogram code used for generating features:

```
def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
   
    # TODO: Compute histograms

    channel_1_hist = np.histogram(channel_1_vals, bins=32, range=(0, 256))
    channel_2_hist = np.histogram(channel_2_vals, bins=32, range=(0, 256))
    channel_3_hist = np.histogram(channel_3_vals, bins=32, range=(0, 256))
 
    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_3_hist[0])).astype(np.float64)

    # Generate random features for demo mode. 
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features)
    return normed_features

def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)
    norm_x_hist = np.histogram(norm_x_vals, bins=32, range=(-1, 1))
    norm_y_hist = np.histogram(norm_y_vals, bins=32, range=(-1, 1))
    norm_z_hist = np.histogram(norm_z_vals, bins=32, range=(-1, 1))
    # TODO: Concatenate and normalize the histograms

    hist_features = np.concatenate((norm_x_hist[0], norm_y_hist[0], norm_z_hist[0])).astype(np.float64)

    # Generate random features for demo mode. 
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features)

    return normed_features
```
The following two images are the confusion matrices generated from the classifier trying to accurately
predict 20 instances of each of the 8 objects. The first one is not normalized and shows how many times of 20 
the trained classifier accurately predicts the correct label. The second image is the same thing but normalized
and shows the percentage of images accurately classified. The eraser is most accurately
predicted at 80% and the glue is least accurately predicted at 50%. 

![alt text][image7]

![alt text][image8]



The following code classifies the clusters with labels and outputs those labels in RViz.

```
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

    # Create dictionary of detected objects
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

    # hard coded number I used to determine current scene of 3
    test_scene_num = Int32()
    test_scene_num.data = 2

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
  
    # Output yaml file with required data
    yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"   
    send_to_yaml(yaml_filename, yaml_dict_list)

```
The following images in order show my results for all three worlds.

![alt text][image1]

![alt text][image2]

![alt text][image3]


I successfully calculated 100% of items (3 of 3) from `pick_list_1.yaml` for `test1.world`, 
100% of items (5 of 5) from `pick_list_2.yaml` for `test2.world` and 
75% of items (6 of 8) from `pick_list_3.yaml` in `test3.world` to meet specifications.

If I had more time I would focus on how to label the two remaining objects in scene 3. The book and the glue were not being identified. I would
look into why that may be. Perhaps adding another feature or looking at the way the objects were clustered and segmented may help or even
trying a different kernel in SVM. Also, if I had more time I would work on the place part of the project as well.

