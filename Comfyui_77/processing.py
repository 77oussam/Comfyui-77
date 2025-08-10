import pyvips
import os

def resize_image(source_path, output_path, target_size, unit='px', dpi=72, export_format='JPG', quality=90):
    """
    Resizes a single image to the specified dimensions and saves it.

    Args:
        source_path (str): Path to the source image.
        output_path (str): Path to save the resized image.
        target_size (int): The target size for the longest edge.
        unit (str): The unit for target_size ('px', 'cm', 'in').
        dpi (int): Dots per inch, for converting non-pixel units.
        export_format (str): The output format ('JPG', 'PNG', 'WEBP').
        quality (int): The compression quality (1-100).
    """
    try:
        # Convert target size to pixels if necessary
        if unit == 'cm':
            target_pixels = int((target_size / 2.54) * dpi)
        elif unit == 'in':
            target_pixels = int(target_size * dpi)
        else:
            target_pixels = int(target_size)

        # Open the image with libvips
        image = pyvips.Image.new_from_file(source_path, access='sequential')

        # Calculate the resize factor
        scale = target_pixels / max(image.width, image.height)

        # Resize the image
        resized_image = image.resize(scale)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the image in the specified format
        if export_format.upper() == 'JPG':
            resized_image.jpegsave(output_path, Q=quality)
        elif export_format.upper() == 'PNG':
            # PNG compression is 0-9, libvips uses a different scale.
            # We can use default settings for simplicity for now.
            resized_image.pngsave(output_path)
        elif export_format.upper() == 'WEBP':
            resized_image.webpsave(output_path, Q=quality)
        else:
            raise ValueError(f"Unsupported format: {export_format}")
            
        return True, f"Successfully resized {os.path.basename(source_path)}"

    except pyvips.Error as e:
        return False, f"Error processing {os.path.basename(source_path)}: {e.message}"
    except Exception as e:
        return False, f"An unexpected error occurred with {os.path.basename(source_path)}: {e}"
