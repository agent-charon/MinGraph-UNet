import cv2
import os
import time

class VideoCapture:
    def __init__(self, output_dir="raw_video_data", camera_index=0):
        self.output_dir = output_dir
        self.camera_index = camera_index
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory for videos: {self.output_dir}")

    def capture_video(self, duration_seconds=30, filename_prefix="mango_video"):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"Error: Could not open video device {self.camera_index}")
            return None

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: # Handle cases where FPS is not reported correctly
            fps = 30.0
            print(f"Warning: Could not get FPS, defaulting to {fps}")


        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.mp4") # Use .mp4 for wider compatibility
        
        # Define the codec and create VideoWriter object
        # FOURCC is a 4-byte code used to specify the video codec.
        # Common codes: 'XVID', 'MJPG', 'MP4V', 'DIVX', 'H264'
        # 'mp4v' is a good choice for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        print(f"Starting video capture: {output_filename}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"Recording for {duration_seconds} seconds. Press 'q' to stop early.")

        start_time = time.time()
        frames_written = 0
        while (time.time() - start_time) < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            
            out.write(frame)
            frames_written += 1
            
            cv2.imshow('Video Capture - Press Q to Stop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capture stopped by user.")
                break
        
        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video capture finished. {frames_written} frames written to {output_filename}")
        return output_filename

if __name__ == '__main__':
    # Example Usage
    # Ensure you have a camera connected. You might need to change camera_index if you have multiple.
    # On Linux, camera_index is often 0 for /dev/video0.
    # On Windows, it's usually 0 for the default camera.
    
    # Check available cameras (optional helper)
    # for i in range(5): # Check first 5 indices
    #     cap_test = cv2.VideoCapture(i)
    #     if cap_test.isOpened():
    #         print(f"Camera index {i} is available.")
    #         cap_test.release()
    #     else:
    #         print(f"Camera index {i} is NOT available.")

    video_capturer = VideoCapture(output_dir="collected_videos", camera_index=0)
    try:
        # Capture a 10-second video
        video_file = video_capturer.capture_video(duration_seconds=10) 
        if video_file:
            print(f"Video saved: {video_file}")
    except Exception as e:
        print(f"An error occurred during video capture: {e}")