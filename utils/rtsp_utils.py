import datetime
from pathlib import Path
from typer import Typer
import cv2
from moviepy.editor import VideoFileClip
from PIL import Image


def get_stream(
    username: str,
    password: str,
    url: str,
    show: bool = True,
    save: bool = False,
    width: int = 1920,
    height: int = 1084,
    downsamp: int = 2,
    timeout: float = 30,
):
    """Get a stream from an IP camera and save it to a file.

    Args:
        username (str): Username of the IP camera.
        password (str): Password of the IP camera.
        url (str): URL of the IP camera.
        show (bool, optional): Show the stream. Defaults to True.
        save (bool, optional): Save the stream to a file. Defaults to False.
        width (int, optional): Width of the stream. Will resize stream accordingly.
         Defaults to 1920.
        height (int, optional): Height of the stream. Will resize stream accordingly.
         Defaults to 1084.
        downsamp (int, optional): Downsampling factor for resizing stream.
         Only used if width or height is not specified. Defaults to 2.
        timeout (float, optional): Timeout for the stream (in minutes). Defaults to 30.

    Returns:
        None

    If save is True, the stream will be saved to a file called "output.mp4".
    """
    # Get stream and set timeout
    stream = cv2.VideoCapture(f"rtsp://{username}:{password}@{url}")

    # Check if the stream was opened successfully
    if not stream.isOpened():
        ValueError("Error opening video stream or file")

    # Get the frame rate and dimensions of the stream
    fps = stream.get(cv2.CAP_PROP_FPS)
    input_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # set timeout period by converting minutes to number of frames
    timeout_frames = int(timeout * 60 * fps)
    print(f"INFO: Timing out in {timeout} minutes ({timeout_frames} frames)")

    # create a function to resize each frame
    if width is None or height is None:
        width = int(input_width // downsamp)
        height = int(input_height // downsamp)
        print(
            "INFO: No width or height given."
            "Resizing stream to {width}x{height} (downsamp={downsamp})"
        )

    resize_frame = lambda in_frame: cv2.resize(in_frame, (width, height))

    if save:
        if fps > 0:
            print(f"INFO: Frame rate of the stream: {fps} fps")
        else:
            fps = 10
            print(
                "WARNING: Can't get the frame rate of the stream (this is a bug)."
                f"Setting it to {fps} fps ..."
            )

        ip = url.split(":")[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{ip}_{timestamp}_{width}x{height}.mp4"

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
        out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))

    # Read until the end of the stream
    frame_count = 0
    while stream.isOpened() and frame_count < timeout_frames:
        # Capture frame-by-frame
        ret, frame = stream.read()

        if not ret:
            print("ERROR: Can't receive frame (stream end?). Exiting ...")
            break

        frame = resize_frame(frame)

        if save:
            out.write(frame)

        if show:
            cv2.imshow(filename, frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # Release the video capture and writer objects and close all windows
    stream.release()
    if save:
        out.release()
        print(f"INFO: Video saved to {filename}")
    cv2.destroyAllWindows()


def get_video_frame(
    video: str,
    time_stamp: float,
    output_name: str = None,
    show: bool = True,
    save: bool = True,
):
    """Get a frame from a video at a given time stamp.

    Args:
        video (str): Path to the video file.
        time_stamp (float): Time stamp (in seconds) of the frame to be extracted.
        output_name (str, optional): Path to the output image file (without extension)
         Defaults to video file + time stamp
        show (bool, optional): Show the frame. Defaults to True.
        save (bool, optional): Save the frame to a file. Defaults to True.

    Returns:
        None

    The output image file will be saved to the current directory.
    """
    clip = VideoFileClip(video)
    frame = clip.get_frame(time_stamp)
    clip.close()

    if output_name is None:
        output_name = Path(video).with_suffix("").as_posix() + f"_{time_stamp}s"

    if save:
        img = Image.fromarray(frame)
        img.save(output_name + ".png", format="PNG")
        img.save(output_name + ".jpg", format="JPEG")

    if show:
        while True:
            cv2.imshow(Path(output_name).name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    # user Typer module to create a command line interface
    app = Typer()
    app.command()(get_stream)
    app.command()(get_video_frame)
    app()
