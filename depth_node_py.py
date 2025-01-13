
import cv2
import depthai as dai
import numpy as np
import time
from collections import deque


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

fpsCounter = FPSCounter()

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


# Connect to device and start pipeline

with dai.Device(pipeline) as device:
    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    fps_deque = deque(maxlen=10)  # Store the last 10 FPS values
    prev_time = time.time() + 5

    while True:
        
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()

        #display fps
        fps = fpsCounter.tick()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        cv2.imshow("disparity", frame)

        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frame)

        if cv2.waitKey(1) == ord('q'):
            break