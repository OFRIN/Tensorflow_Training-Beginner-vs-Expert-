
import cv2
import imageio

video_path = './results/without_Thread_GPU.mp4'

frames = []

video = cv2.VideoCapture(video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))

for frame_index in range(1 * fps, 10 * fps, 5):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    ret, frame = video.read()
    frames.append(frame[..., ::-1])
    # frames.append(frame)

imageio.mimsave(video_path.replace('.mp4', '.gif'), frames)
