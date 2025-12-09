import os
from tkinter import Tk, filedialog, messagebox
from PIL import Image, ImageOps

def invert_image(input_path, output_path=None):
    # Open image
    img = Image.open(input_path).convert("RGBA")  # keep alpha if present

    # Split out alpha channel
    r, g, b, a = img.split()

    # Invert RGB channels
    rgb_image = Image.merge("RGB", (r, g, b))
    inverted_rgb = ImageOps.invert(rgb_image)

    # Put alpha back
    r2, g2, b2 = inverted_rgb.split()
    inverted_img = Image.merge("RGBA", (r2, g2, b2, a))

    # Decide output path
    if output_path is None:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_inverted{ext}"

    inverted_img.save(output_path)
    return output_path


def main():
    # Hide main Tk window
    root = Tk()
    root.withdraw()

    # Ask user to pick an image file
    file_path = filedialog.askopenfilename(
        title="Select an image to invert",
        filetypes=[
            ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff",
                             "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.GIF", "*.TIFF")),
            ("All files", "*.*"),
        ],
    )

    if not file_path:
        # User cancelled
        messagebox.showinfo("No file selected", "No file was selected. Exiting.")
        return

    try:
        output_path = invert_image(file_path)
        messagebox.showinfo("Success", f"Inverted image saved to:\n{output_path}")
        print(f"Inverted image saved to: {output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to invert image:\n{e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
