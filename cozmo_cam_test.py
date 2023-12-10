import cozmo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import asyncio
from statistics import median

# This list will hold the last 10 deviation values
deviations = []

def median_filter(new_deviation, deviations, window_size=10):
    # Add the new deviation to the list of deviations
    deviations.append(new_deviation)
    # If we have more than the window size, remove the oldest deviation
    if len(deviations) > window_size:
        deviations.pop(0)
    # Calculate the median of the deviations
    if deviations:
        return median(deviations)
    else:
        return 0
def find_line_center(contours):
    # Find the largest contour and its center
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
    else:
        cX = None
    return cX

def cozmo_program(robot: cozmo.robot.Robot):
    # Set Cozmo's camera to stream in color
    robot.camera.color_image_enabled = True

    # Define your line color bounds in HSV format
    line_color_lower = np.array([160, 100, 100])  # Replace with actual values
    line_color_upper = np.array([180, 200, 200])  # Replace with actual values

     #async def process_frame():
    robot.set_head_angle(cozmo.util.degrees(-25)).wait_for_completed()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    out = cv2.VideoWriter('cozmo_output.avi', fourcc, 20.0, (260, 480))  # Filename, codec, fps, frame size

    while True:
        deviations = []
        while len(deviations) < 5:
            # Get the latest image from Cozmo's camera
            robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=5)
            pil_image = robot.world.latest_image.raw_image.convert("RGB")

            # Convert PIL Image to numpy array
            image = np.array(pil_image)
            image_np = image[50:image.shape[0]-60, 20:280]
            #print(image_np.shape)

            # Convert to HSV color space
            #hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

            # Create a mask for the line color
            #mask = cv2.inRange(hsv, line_color_lower, line_color_upper)

            # Find contours
            # contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_center = image_np.shape[1] // 2
            #print(image_np.shape)
            bw = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            threshold_value = 30

            # Apply the threshold to make white whiter and black blacker
            _, bw = cv2.threshold(bw, threshold_value, 255, cv2.THRESH_BINARY_INV)
            #print(bw[0,1])
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours: 
                line_center = find_line_center(contours)
                #print(image_center, line_center)

                # Calculate deviation from the center of the image if a line center was found
                if line_center is not None:
                    dev = line_center - image_center
                # else: 
                #     dev = float('inf')
                # Add the new deviation to the list of deviations
                    deviations.append(dev)

        deviation = median(deviations)

        # pil_image_draw = ImageDraw.Draw(image_np)
        # font = ImageFont.load_default()
        # text = f"Deviation: {deviation}"
        # pil_image_draw.text((image_np.shape[1] - 100, 10), text, (255, 255, 255), font=font)
        cv2.putText(bw, f"Deviation: {deviation}", (image_np.shape[1] - 100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 110, 100), 1)
        cv2.imshow("Cozmo's View", bw)
        out.write(bw)

        # Create an empty image of the same shape as the original
        # contourImage = np.zeros_like(mask)

        # # Draw the contours
        # # Define a minimum area threshold for contours
        # min_contour_area = -5 # Adjust this value based on your requirements

        # Filter contours by area
        # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        
        # Display the image #cv2.drawContours(contourImage, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the Video Writer
    out.release()
    cv2.destroyAllWindows()

    # Start processing frames
    #robot.loop.create_task(process_frame())

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=False)
