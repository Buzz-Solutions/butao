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
    width: int = None,
    height: int = None,
    downsamp: int = 1,
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
        height (int, optional): Height of the stream. Will resize stream accordingly.
        downsamp (int, optional): Downsampling factor for resizing stream.
         Only used if width or height is not specified.
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

    if width is not None and height is not None:
        print(f"Resizing stream from {input_width}x{input_height} to {width}x{height}")
    elif downsamp != 1:
        width = int(input_width // downsamp)
        height = int(input_height // downsamp)
        print(
            f"INFO: No width or height given."
            f"Resizing stream to {width}x{height} (downsamp={downsamp})"
        )
    else:
        width = input_width
        height = input_height
        print(f"Image dimensions {width}x{height}")

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
    save: str = "jpg",
    repeat_frames: int = 1,
):
    """Get a frame from a video at a given time stamp.

    Args:
        video (str): Path to the video file.
        time_stamp (float): Time stamp (in seconds) of the frame to be extracted.
        output_name (str, optional): Path to the output image file (without extension).
         Defaults to video file + time stamp
        show (bool, optional): Show the frame. Defaults to True.
        save (str, optional): Save the frame to a file. Defaults to jpg.
         Options are jpg, png, and mp4
        repeat_frames (int, optional): Number of times to repeat the frame in the
         created video. Only enabled if save set to mp4. Defaults to 1.

    Returns:
        None

    The output image file will be saved to the current directory.
    """
    clip = VideoFileClip(video)
    frame = clip.get_frame(time_stamp)
    clip.close()

    if output_name is None:
        output_name = Path(video).with_suffix("").as_posix() + f"_{time_stamp}s"

    if save and (save == "jpg" or save == "png"):
        img = Image.fromarray(frame)
        img.save(output_name + ".png", format="PNG")
        img.save(output_name + ".jpg", format="JPEG")

    elif save and save == "mp4":
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
        out = cv2.VideoWriter(output_name + ".mp4", fourcc, clip.fps, (width, height))

        # Write the frame to the video file n times
        for _ in range(repeat_frames):
            out.write(frame)

        out.release()

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
