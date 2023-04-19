from PIL import Image


def resample_image(
    image_path: str,
    factor: float = 1,
    save: str = None,
    width: int = None,
    height: int = None,
):
    """
    Resample an image either by a given factor or by a given width and height

    :param image_path: path to the input image
    :param factor: factor to resample the image (e.g. 0.5 means downsample by half in
     each dimension)
    :param save: if not None, save the resampled image to the given path
    :param width: if not None, resize the image to the given width (height must be set)
     (overrides factor)
    :param height: if not None, resize the image to the given height (width must be set)
     (overrides factor)

    :return: the downsampled image as a PIL Image
    """
    with Image.open(image_path) as im:
        if width is not None and height is not None:
            new_size = (width, height)
        else:
            new_size = (im.width // factor, im.height // factor)

        print(f"INFO: Resizing image from {im.size} to {new_size}")
        im_resized = im.resize((width, height))

        if save:
            im_resized.save(save, format="JPEG")

    return im_resized
