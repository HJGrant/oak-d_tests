
import cv2
import depthai as dai
import numpy as np
import time
from collections import deque
import open3d as o3d
import open3d.core as o3c


class FPSCounter:
    def __init__(self):
        self.frameCount = 0
        self.fps = 0
        self.startTime = time.time()

    def tick(self):
        self.frameCount += 1
        if self.frameCount % 10 == 0:
            elapsedTime = time.time() - self.startTime
            self.fps = self.frameCount / elapsedTime
            self.frameCount = 0
            self.startTime = time.time()
        return self.fps

def setup_oak_d():
    # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
    extended_disparity = False

    # Better accuracy for longer distance, fractional disparity 32-levels:
    subpixel = False

    # Better handling for occlusions:
    lr_check = False

    # Create pipeline
    pipeline = dai.Pipeline()
    #device_info = dai.DeviceInfo("169.254.1.222")

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("disparity")

    resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

    # Properties
    monoLeft.setResolution(resolution)
    monoLeft.setCamera("left")
    monoRight.setResolution(resolution)
    monoRight.setCamera("right")

    monoLeft.setFps(110.0)
    monoRight.setFps(110.0)

    # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    depth.setLeftRightCheck(lr_check)
    depth.setExtendedDisparity(extended_disparity)
    depth.setSubpixel(subpixel)
    #depth.setInputResolution(1280, 720)


    # Linking
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.disparity.link(xout.input)

    return pipeline, depth

def get_oak_d_calib(device):
    calib = device.readCalibration()

    # Get intrinsics for left camera
    left_intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.LEFT)
    fx, fy, cx, cy = left_intrinsics[0][0], left_intrinsics[1][1], left_intrinsics[0][2], left_intrinsics[1][2]

    # Create Open3D intrinsic matrix
    intrinsic_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)

    # Convert to Open3D Tensor
    intrinsic_tensor = o3d.core.Tensor(intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # Get extrinsics (rotation and translation) for left to right cameras
    left_to_right_extrinsics = np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
    extrinsic_tensor = o3d.core.Tensor(left_to_right_extrinsics, dtype=o3d.core.Dtype.Float64)

    # Get baseline (optional, but useful for depth)
    baseline = calib.getBaselineDistance(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT)

    return intrinsic_tensor, extrinsic_tensor, baseline

def setup_voxel_grid():
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=['tsdf', 'weight'],
        attr_dtypes=[o3c.float32, o3c.float32],
        attr_channels=[[1], [1]],  # Must be lists of lists
        voxel_size=3.0 / 512,
        block_resolution=16,
        block_count=50000,
        device=o3c.Device('CPU:0')  # Correct device specification
    )
    return vbg

def compute_voxel_grid(depth, depth_intrinsic, extrinsic, depth_scale, depth_max):
        #integrate voxel grid
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, depth_intrinsic, extrinsic, depth_scale,
            depth_max)
            
        vbg.integrate(frustum_block_coords, depth, depth_intrinsic,
                        extrinsic, depth_scale, depth_max)        
        return vbg

def compute_depth_as_image(intrinsics: o3d.core.Tensor, disparity_in_pixels, baseline: float) -> o3d.geometry.Image:

    disparity_in_pixels[disparity_in_pixels==0] = 1

    fx = intrinsics[0, 0].item()  # Convert to scalar for computation

    # Compute depth: Z = fx * baseline / disparity
    depth_map = fx * baseline / disparity_in_pixels

    # Create Open3D Image from the NumPy array
    depth_image = o3d.t.geometry.Image(depth_map.astype(np.float32))

    return depth_image

fpsCounter = FPSCounter()

pipeline, depth = setup_oak_d()

global vbg
vbg = setup_voxel_grid()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    ##VISUALIZATION###
    isRunning = True
    def key_callback(vis, action, mods):
        global isRunning
        if action == 0:
            isRunning = False
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_action_callback(81, key_callback)
    coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0,0,0])
    vis.add_geometry(coordinateFrame)

    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    #q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    intrinsics, extrinsics, baseline = get_oak_d_calib(device)

    fps_deque = deque(maxlen=10)  # Store the last 10 FPS values
    prev_time = time.time() + 5

    i=0
    while True:
        
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()

        #display fps
        fps = fpsCounter.tick()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        cv2.imshow("disparity", frame)

        frame_color = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frame_color)

        depth_map = compute_depth_as_image(intrinsics, frame, baseline)

        compute_voxel_grid(depth_map, intrinsics, extrinsics, 1000.0, 3.0)

        if i > 100: 
            pcd = vbg.extract_point_cloud()
            print(type(pcd))
            #o3d.visualization.draw([pcd])
            vis.add_geometry(pcd)

        if i>101:
            vis.update_geometry(pcd)
                        

        i+=1
        print(i)

        if cv2.waitKey(1) == ord('q'):
            break