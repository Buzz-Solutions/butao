from typer import Typer
import cv2


def get_stream(
    username: str,
    password: str,
    url: str,
    show: bool = True,
    save: bool = False,
    width: int = 640,
    height: int = 480,
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
         Defaults to 640.
        height (int, optional): Height of the stream. Will resize stream accordingly.
         Defaults to 480.
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
    if save:
        fps = stream.get(cv2.CAP_PROP_FPS)
        input_width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if fps > 0:
            print(f"INFO: Frame rate of the stream: {fps} fps")
        else:
            fps = 10
            print(
                "WARNING: Can't get the frame rate of the stream (this is a bug)."
                f"Setting it to {fps} fps ..."
            )

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (int(width), int(height)))

    # set timeout period by converting minutes to number of frames
    timeout_frames = int(timeout * 60 * fps)
    print(f"INFO: Timing out in {timeout} minutes ({timeout_frames} frames)")

    # create a function to resize each frame
    if width is not None and height is not None:
        resize_frame = lambda in_frame: cv2.resize(in_frame, (width, height))
    else:
        resize_frame = lambda in_frame: cv2.resize(
            in_frame, (input_width // downsamp, input_height // downsamp)
        )

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
            cv2.imshow("Frame", frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # Release the video capture and writer objects and close all windows
    stream.release()
    if save:
        out.release()
        print("INFO: Video saved to output.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # user Typer module to create a command line interface
    app = Typer()
    app.command()(get_stream)
    app()
