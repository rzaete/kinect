import argparse
import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt

date = "5_4"

class ViewerWithCallback:

	def __init__(self, config, device, align_depth_to_color):
		self.flag_exit = False
		self.align_depth_to_color = align_depth_to_color

		self.sensor = o3d.io.AzureKinectSensor(config)
		if not self.sensor.connect(device):
			raise RuntimeError('Failed to connect to sensor')

	def escape_callback(self, vis):
		self.flag_exit = True
		return False

	def record(self, max_frame_number):
		glfw_key_escape = 256
		vis = o3d.visualization.VisualizerWithKeyCallback()
		vis.register_key_callback(glfw_key_escape, self.escape_callback)
		vis.create_window('viewer', 1920, 540)
		print("Sensor initialized. Press [ESC] to exit.")
		
		color_dir = "color"
		depth_dir = "depth"
		if not os.path.exists(color_dir):
			os.makedirs(color_dir)
		if not os.path.exists(depth_dir):
			os.makedirs(depth_dir)
			
		vis_geometry_added = False
		frame_number = 0
		rgbd_list = []
		while frame_number < max_frame_number:
			rgbd = self.sensor.capture_frame(self.align_depth_to_color)
			if rgbd is None:
				print("image is none")
				continue
			
			rgbd_list.append([np.copy(np.asarray(rgbd.color)), np.copy(np.asarray(rgbd.depth))])

			if not vis_geometry_added:
				vis.add_geometry(rgbd)
				vis_geometry_added = True

			vis.update_geometry(rgbd)
			vis.poll_events()
			vis.update_renderer()
			frame_number += 1

		for i in range(len(rgbd_list)):
			[color, depth] = rgbd_list[i]
			color_image = o3d.geometry.Image(color)
			depth_image = o3d.geometry.Image(depth)
			filename = str(i) + ".png"
			color_file_path = os.path.join(color_dir, filename)
			depth_file_path = os.path.join(depth_dir, filename)
			o3d.io.write_image(color_file_path, color_image, quality=100)
			o3d.io.write_image(depth_file_path, depth_image, quality=100)

	def run(self):
		glfw_key_escape = 256
		vis = o3d.visualization.VisualizerWithKeyCallback()
		vis.register_key_callback(glfw_key_escape, self.escape_callback)
		vis.create_window('viewer', 1920, 540)
		print("Sensor initialized. Press [ESC] to exit.")

		vis_geometry_added = False
		switch = False
		while not self.flag_exit:
			rgbd = self.sensor.capture_frame(self.align_depth_to_color)
			if rgbd is None:
				print("image is none")
				continue

			# if not switch:
			#   # d = np.asarray(rgbd.depth).astype(np.float32)
			#   # print("type:", d.dtype)
			#   # rgbd.depth = o3d.geometry.Image(np.asarray(rgbd.depth).astype(np.float32))
			#   print("color:", rgbd.color)
			#   print(np.asarray(rgbd.color).dtype)
			#   print("depth:", rgbd.depth)
			#   print(np.asarray(rgbd.depth).dtype)
			#   switch = True
			rgbd.depth = o3d.geometry.Image(np.asarray(rgbd.depth).astype(np.float32))
			pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
			# print(pcd)
			vis.clear_geometries()
			if not vis_geometry_added:
				vis.add_geometry(pcd, reset_bounding_box=True)
				vis_geometry_added = True
			else:
				vis.add_geometry(pcd, reset_bounding_box=False)

			vis.update_geometry(pcd)
			vis.poll_events()
			vis.update_renderer()

class ViewerWithCallback2:

	def __init__(self):
		self.flag_exit = False

	def escape_callback(self, vis):
		self.flag_exit = True
		return False

	def run(self, pcd_list):
		glfw_key_escape = 256
		vis = o3d.visualization.VisualizerWithKeyCallback()
		vis.register_key_callback(glfw_key_escape, self.escape_callback)
		vis.create_window('viewer', 1920, 540)
		print("Sensor initialized. Press [ESC] to exit.")

		vis_geometry_added = False
		switch = False
		while not self.flag_exit:
			for [pcd_main, pcd_secondary] in pcd_list:
				vis.clear_geometries()
				if not vis_geometry_added:
					vis.add_geometry(pcd_main, reset_bounding_box = True)
					vis.add_geometry(pcd_secondary, reset_bounding_box = True)
					vis_geometry_added = True
				else:
					vis.add_geometry(pcd_main, reset_bounding_box = False)
					vis.add_geometry(pcd_secondary, reset_bounding_box = False)

				vis.update_geometry(pcd_main)
				vis.update_geometry(pcd_secondary)
				vis.poll_events()
				vis.update_renderer()

	def run2(self, pcd_list):
		glfw_key_escape = 256
		vis = o3d.visualization.VisualizerWithKeyCallback()
		vis.register_key_callback(glfw_key_escape, self.escape_callback)
		vis.create_window('viewer', 1920, 540)
		print("Sensor initialized. Press [ESC] to exit.")

		vis_geometry_added = False
		switch = False
		while not self.flag_exit:
			for pcd in pcd_list:
				vis.clear_geometries()
				if not vis_geometry_added:
					vis.add_geometry(pcd, reset_bounding_box = True)
					vis_geometry_added = True
				else:
					vis.add_geometry(pcd, reset_bounding_box = False)

				vis.update_geometry(pcd)
				vis.poll_events()
				vis.update_renderer()       

class ViewerWithCallback3:

	def __init__(self):
		self.flag_exit = False

	def escape_callback(self, vis):
		self.flag_exit = True
		return False

	def run(self, mesh_list):
		glfw_key_escape = 256
		vis = o3d.visualization.VisualizerWithKeyCallback()
		vis.register_key_callback(glfw_key_escape, self.escape_callback)
		vis.create_window('viewer', 1920, 540)
		print("Sensor initialized. Press [ESC] to exit.")

		vis_geometry_added = False
		switch = False
		while not self.flag_exit:
			for mesh in mesh_list:
				vis.clear_geometries()
				if not vis_geometry_added:
					vis.add_geometry(mesh, reset_bounding_box = True)
					vis_geometry_added = True
				else:
					vis.add_geometry(mesh, reset_bounding_box = False)

				vis.update_geometry(mesh)
				vis.poll_events()
				vis.update_renderer()

def test_kinect():
	o3d.io.AzureKinectSensor.list_devices()
	# config = o3d.io.AzureKinectSensorConfig()
	config = o3d.io.read_azure_kinect_sensor_config("config.json")
	device = 0
	align_depth_to_color = True
	v = ViewerWithCallback(config, device, align_depth_to_color)
	v.run()

def test_mesh():
	pcd = o3d.io.read_point_cloud("cloud_bin_0.pcd")
	bpa = True
	poisson = False

	if bpa:
		distances = pcd.compute_nearest_neighbor_distance()
		avg_dist = np.mean(distances)
		radius = 3 * avg_dist
		bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
		dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
		dec_mesh.remove_degenerate_triangles()
		dec_mesh.remove_duplicated_triangles()
		dec_mesh.remove_duplicated_vertices()
		dec_mesh.remove_non_manifold_edges()
		o3d.visualization.draw_geometries([dec_mesh])

	if poisson:
		poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, width=0, scale=1.1, linear_fit=False)[0]
		bbox = pcd.get_axis_aligned_bounding_box()
		p_mesh_crop = poisson_mesh.crop(bbox)
		dec_mesh2 = p_mesh_crop.simplify_quadric_decimation(100000)
		o3d.visualization.draw_geometries([dec_mesh2])

def create_dataset():
	o3d.io.AzureKinectSensor.list_devices()
	# config = o3d.io.AzureKinectSensorConfig()
	config = o3d.io.read_azure_kinect_sensor_config("config.json")
	device = 0
	align_depth_to_color = True
	v = ViewerWithCallback(config, device, align_depth_to_color)
	v.record(45)

def create_point_cloud():
	pass

def test_read_point_cloud(prefix = "pc_raw_"):
	main_camera_dir = os.path.join("experiments", "two_sensor", "main_camera")
	secondary_camera_dir = os.path.join("experiments", "two_sensor", "secondary_camera")
	
	i = 5
	main_pc_path = os.path.join(main_camera_dir, prefix + str(i) + ".ply")
	secondary_pc_path = os.path.join(secondary_camera_dir, prefix + str(i) + ".ply")
	main_pcd = o3d.io.read_point_cloud(main_pc_path)
	main_pcd.paint_uniform_color([1, 0.706, 0])     # Orange
	secondary_pcd = o3d.io.read_point_cloud(secondary_pc_path)
	secondary_pcd.paint_uniform_color([0, 0.651, 0.929])        # Blue

	o3d.visualization.draw_geometries([main_pcd, secondary_pcd])

def test_read_multiple_point_cloud(prefix = "pc_raw_"):
	main_camera_dir = os.path.join("experiments", "two_sensor", "main_camera")
	secondary_camera_dir = os.path.join("experiments", "two_sensor", "secondary_camera")
	pcd_list = []
	
	for i in range(6):
		main_pc_path = os.path.join(main_camera_dir, prefix + str(i) + ".ply")
		secondary_pc_path = os.path.join(secondary_camera_dir, prefix + str(i) + ".ply")
		main_pcd = o3d.io.read_point_cloud(main_pc_path)
		main_pcd.paint_uniform_color([1, 0.706, 0])     # Orange
		secondary_pcd = o3d.io.read_point_cloud(secondary_pc_path)
		secondary_pcd.paint_uniform_color([0, 0.651, 0.929])        # Blue
		pcd_list.append([main_pcd, secondary_pcd])

	v = ViewerWithCallback2()
	v.run(pcd_list)

def test_my_mesh():
	i = 44
	color_dir = os.path.join("experiments", "mesh_reconstruction", "color")
	depth_dir = os.path.join("experiments", "mesh_reconstruction", "depth")
	color_file_path = os.path.join(color_dir, str(i) + ".png")
	depth_file_path = os.path.join(depth_dir, str(i) + ".png")

	color_image = o3d.io.read_image(color_file_path)
	depth_image = o3d.io.read_image(depth_file_path)
	rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
	rgbd.depth = o3d.geometry.Image(np.asarray(rgbd.depth).astype(np.float32))
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
	downpcd = pcd.voxel_down_sample(voxel_size=0.001)
	downpcd.estimate_normals()
	# o3d.visualization.draw_geometries([downpcd])
	bpa = True
	poisson = False

	if bpa:
		mesh = create_mesh_from_point_cloud(downpcd)
		o3d.visualization.draw_geometries([mesh])

def test_multiple_mesh():
	pass

def write_combined_point_clouds(prefix = "pc_transformed_"):
	number_of_files = 45
	main_camera_dir = os.path.join("experiments", "two_sensor", date, "main_camera")
	secondary_camera_dir = os.path.join("experiments", "two_sensor", date, "secondary_camera")
	combined_dir = os.path.join("experiments", "two_sensor", date, "combined")
	vol = o3d.visualization.read_selection_polygon_volume("cropped_1.json")
	
	for i in range(number_of_files):
		main_pc_path = os.path.join(main_camera_dir, prefix + str(i) + ".ply")
		secondary_pc_path = os.path.join(secondary_camera_dir, prefix + str(i) + ".ply")
		main_pcd = o3d.io.read_point_cloud(main_pc_path)
		main_pcd = vol.crop_point_cloud(main_pcd)
		# main_pcd.paint_uniform_color([1, 0.706, 0])       # Orange
		secondary_pcd = o3d.io.read_point_cloud(secondary_pc_path)
		secondary_pcd = vol.crop_point_cloud(secondary_pcd)
		combined_pcd = main_pcd + secondary_pcd
		down_sampled_pcd = combined_pcd.voxel_down_sample(voxel_size=0.5)
		combined_path = os.path.join(combined_dir, prefix + str(i) + '.ply')
		o3d.io.write_point_cloud(combined_path, down_sampled_pcd, write_ascii=True)
		# secondary_pcd.paint_uniform_color([0, 0.651, 0.929])      # Blue
		# pcd_list.append(down_sampled_pcd)

def read_point_clouds(prefix = "pc_transformed_"):
	number_of_files = 45
	pcd_list = []
	combined_dir = os.path.join("experiments", "two_sensor", date, "combined")
	
	for i in range(number_of_files):
		combined_path = os.path.join(combined_dir, prefix + str(i) + '.ply')
		pcd = o3d.io.read_point_cloud(combined_path)
		pcd_list.append(pcd)

	return pcd_list


def create_poisson_mesh_from_point_cloud(pcd):
	poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
	bbox = pcd.get_axis_aligned_bounding_box()
	p_mesh_crop = poisson_mesh.crop(bbox)
	dec_mesh = p_mesh_crop.simplify_quadric_decimation(100000)
	return dec_mesh

def create_mesh_from_point_cloud(pcd):
	# print("create mesh from pcd")
	# distances = pcd.compute_nearest_neighbor_distance()
	# avg_dist = np.mean(distances)
	# radius = 3 * avg_dist
	radius = 8
	# print("bpa")
	bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
	# print("bpa finished")
	# dec_mesh = bpa_mesh.simplify_quadric_decimation(500000)
	# dec_mesh.remove_degenerate_triangles()
	# dec_mesh.remove_duplicated_triangles()
	# dec_mesh.remove_duplicated_vertices()
	# dec_mesh.remove_non_manifold_edges()
	return bpa_mesh

def write_mesh_to_file(mesh, bpa, index):
	method = "bpa"
	if not bpa:
		method = "poisson"
	mesh_path = os.path.join("experiments", "mesh_reconstruction", date, method, "mesh_" + str(index) + ".ply")
	o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=True)

def reconstruct_mesh():
	bpa = False
	print("Start reading point clouds.")
	pcd_list = read_point_clouds()
	print(len(pcd_list), "point clouds opened.")
	# v = ViewerWithCallback2()
	# v.run2(pcd_list)
	mesh_list = []

	for i, pcd in enumerate(pcd_list):
		print("mesh", i)
		pcd.estimate_normals()
		res = pcd.orient_normals_towards_camera_location()
		if not res:
			print("cannot orient noramls")
		pcd = pcd.uniform_down_sample(5)
		if bpa:
			mesh = create_mesh_from_point_cloud(pcd)
		else:
			mesh = create_poisson_mesh_from_point_cloud(pcd)
		write_mesh_to_file(mesh, bpa, i)
		mesh_list.append(mesh)

	v = ViewerWithCallback3()
	v.run(mesh_list)


def rename_files():
	main_camera_dir = os.path.join("experiments", "two_sensor", "main_camera")
	secondary_camera_dir = os.path.join("experiments", "two_sensor", "secondary_camera")
	
	for f in os.listdir(main_camera_dir):
		file_path = os.path.join(main_camera_dir, f)
		if os.path.isfile(file_path):
			new_name = f.replace('camera1_', '')
			os.rename(file_path, os.path.join(main_camera_dir, new_name))

	for f in os.listdir(secondary_camera_dir):
		file_path = os.path.join(secondary_camera_dir, f)
		if os.path.isfile(file_path):
			new_name = f.replace('camera2_', '')
			os.rename(file_path, os.path.join(secondary_camera_dir, new_name))


def main():
	# test_kinect()
	# test_read_point_cloud(prefix = "pc_transformed_")
	# test_read_multiple_point_cloud(prefix = "pc_transformed_")
	# test_mesh()
	# create_dataset()
	# create_point_cloud()
	# rename_files()
	# test_my_mesh()
	reconstruct_mesh()
	# write_combined_point_clouds()


if __name__ == "__main__":
	main()