import cv2
import os

class FrameExtractor:
    def __init__(self, output_dir="extracted_frames"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory for frames: {self.output_dir}")

    def extract_frames(self, video_path, frame_interval=1, output_format="png"):
        """
        Extracts frames from a video file.

        Args:
            video_path (str): Path to the video file.
            frame_interval (int): Extract every Nth frame. 1 means all frames.
            output_format (str): "png" or "jpg".
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        frame_output_subdir = os.path.join(self.output_dir, video_filename)
        os.makedirs(frame_output_subdir, exist_ok=True)

        frame_count = 0
        extracted_count = 0

        print(f"Extracting frames from {video_path} to {frame_output_subdir}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(frame_output_subdir, f"frame_{extracted_count:05d}.{output_format}")
                cv2.imwrite(frame_filename, frame)
                extracted_count += 1
            
            frame_count += 1
            
            if frame_count % 100 == 0: # Progress update
                 print(f"Processed {frame_count} frames, extracted {extracted_count}...")

        cap.release()
        print(f"Finished extracting frames. Total {extracted_count} frames saved to {frame_output_subdir}.")

if __name__ == '__main__':
    # Example Usage
    # First, ensure you have a video file, e.g., from video_capture.py
    # Suppose a video was saved at "collected_videos/mango_video_20231027_100000.mp4"
    
    extractor = FrameExtractor(output_dir="dataset/raw_images") # Store in a more dataset-like structure
    
    # Create a dummy video for testing if you don't have one
    dummy_video_path = "dummy_video.mp4"
    if not os.path.exists(dummy_video_path):
        print(f"Creating a dummy video: {dummy_video_path} for testing frame_extractor.py")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_dummy = cv2.VideoWriter(dummy_video_path, fourcc, 20.0, (640, 480))
        for _ in range(60): # 3 seconds of video
            dummy_frame = cv2.UMat(480, 640, cv2.CV_8UC3) # Black frame
            dummy_frame[:] = (0,0,0)
            cv2.putText(dummy_frame, f"Frame {_}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out_dummy.write(dummy_frame)
        out_dummy.release()
        print("Dummy video created.")

    # Example: Extract every 5th frame from the dummy video
    extractor.extract_frames(dummy_video_path, frame_interval=5, output_format="jpg")
    
    # Example for a real video (replace with your actual video path)
    # video_to_process = "collected_videos/your_mango_video.mp4" 
    # if os.path.exists(video_to_process):
    #     extractor.extract_frames(video_to_process, frame_interval=10, output_format="png")
    # else:
    #     print(f"Video {video_to_process} not found. Skipping.")