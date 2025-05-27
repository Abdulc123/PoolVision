from ultralytics import YOLO
import sounddevice as sd

print(sd.query_devices())

# # Load YOLOv8 nano model
# model = YOLO('yolov8n.pt')

# # Run inference on a sample image (Ultralytics test image)
# results = model('https://ultralytics.com/images/bus.jpg')

# # Print results summary
# print(results)

# # Display the image with detections
# results[0].show()


"""
conda init cmd.exe
conda activate yolo (once base is next to file path), after running yolo will be next to it
"""