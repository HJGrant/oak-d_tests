import cv2
import depthai as dai
import time
from collections import deque
import numpy as np

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

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
FPS_COLOR=30.0
FPS_DEPTH=110.0

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
camRgb = pipeline.create(dai.node.Camera)
disparityOut = pipeline.create(dai.node.XLinkOut)
rgbOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
disparityOut.setStreamName("disp")

# Set up RGB camera
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setSize(680, 400)
camRgb.setFps(FPS_COLOR)

try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise

# Set up mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setCamera("right")
monoLeft.setFps(FPS_DEPTH)
monoRight.setFps(FPS_DEPTH)

# Set up stereo depth configuration
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(False)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(disparityOut.input)
camRgb.video.link(rgbOut.input)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    dispQueue = device.getOutputQueue(name="disp", maxSize=4, blocking=False)

    while True:
        frameRgb = rgbQueue.get().getCvFrame()
        frameDisp = dispQueue.get().getFrame()

        # Display FPS
        fps = fpsCounter.tick()
        cv2.putText(frameDisp, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Normalize for better visualization
        frameDisp = (frameDisp * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        cv2.imshow("disparity", frameDisp)

        frameDisp_color = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frameDisp_color)
        cv2.imshow("RGB", frameRgb)

        if cv2.waitKey(1) == ord('q'):
            break
