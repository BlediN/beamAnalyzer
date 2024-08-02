import cv2
import csv
import numpy as np
from tkinter import Tk, Label, Button, Canvas, Scrollbar, Frame, filedialog, ttk, Entry
from PIL import Image, ImageTk
import os
import pickle

uploaded_filename = None
scale_pixels_per_micron = 10

def process_image(image_path):
    global scale_pixels_per_micron
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Convert the image to binary using thresholding
    _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    min_contour_area = 50
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    
    # Draw the contours on a copy of the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
    
    # Process image to calculate diameter, area, and FWHM
    diameter, area, fwhm_x, fwhm_y = calculate_beam_properties(image, contours)

    # Use edge detection to better identify the beam rays
    edges = cv2.Canny(image, 50, 150)
    
    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    # Create an image to draw the lines
    line_image = np.zeros_like(image)
    
    # If lines are detected, draw them
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Find contours on the line image
    contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    
    # Draw the contours on a copy of the original image
    contour_image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image_with_lines, filtered_contours, -1, (0, 255, 0), 2)
    
    # Find the bounding boxes of the contours
    bounding_boxes_with_lines = [cv2.boundingRect(contour) for contour in filtered_contours]
    
    # Sort the bounding boxes by the x-coordinate
    bounding_boxes_with_lines.sort(key=lambda x: x[0])
    
    # Calculate the distances between the centers of the bounding boxes
    distances_with_lines = []
    for i in range(len(bounding_boxes_with_lines) - 1):
        x1, y1, w1, h1 = bounding_boxes_with_lines[i]
        x2, y2, w2, h2 = bounding_boxes_with_lines[i + 1]
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        distances_with_lines.append(distance)
    
    # Calculate the average distance
    average_distance_with_lines = np.mean(distances_with_lines) if distances_with_lines else 0
    
    # Convert distances to microns
    distances_in_microns_with_lines = [distance / scale_pixels_per_micron for distance in distances_with_lines]
    average_distance_microns_with_lines = np.mean(distances_in_microns_with_lines) if distances_in_microns_with_lines else 0
    
    return (image, contour_image, contour_image_with_lines, distances_with_lines, 
            average_distance_with_lines, distances_in_microns_with_lines, average_distance_microns_with_lines,
            diameter, area, fwhm_x, fwhm_y)

def calculate_beam_properties(image, contours):
    if not contours:
        raise ValueError("No contours found in the image.")
    
    # Assuming the largest contour is the beam dot
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the minimum enclosing circle and the area
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    area = np.pi * (radius ** 2)
    
    # Calculate the FWHM
    fwhm_x, fwhm_y = calculate_fwhm(image, center)
    
    return radius * 2, area, fwhm_x, fwhm_y

def calculate_fwhm(image, center):
    # Extract intensity profile along horizontal and vertical lines through the center
    x_profile = image[center[1], :]
    y_profile = image[:, center[0]]
    
    # Calculate half maximum intensity
    half_max_x = np.max(x_profile) / 2
    half_max_y = np.max(y_profile) / 2
    
    # Calculate FWHM for x_profile
    indices_x = np.where(x_profile >= half_max_x)[0]
    fwhm_x = indices_x[-1] - indices_x[0] if indices_x.size > 1 else 0
    
    # Calculate FWHM for y_profile
    indices_y = np.where(y_profile >= half_max_y)[0]
    fwhm_y = indices_y[-1] - indices_y[0] if indices_y.size > 1 else 0
    
    return fwhm_x, fwhm_y

def open_file_dialog():
    global uploaded_filename
    file_path = filedialog.askopenfilename()
    if file_path:
        uploaded_filename = file_path
        update_filename_display(file_path)

def update_filename_display(file_path):
    filename = os.path.basename(file_path)
    filename_label.config(text=f"Filename: {filename}")

def update_image_display(original_image, contour_image, contour_image_with_lines):
    # Convert images to PhotoImage format
    original_photo = ImageTk.PhotoImage(Image.fromarray(original_image))
    contour_photo = ImageTk.PhotoImage(Image.fromarray(contour_image))
    contour_photo_with_lines = ImageTk.PhotoImage(Image.fromarray(contour_image_with_lines))
    
    # Update the labels with the images
    original_image_label.config(image=original_photo)
    original_image_label.image = original_photo
    contour_image_label.config(image=contour_photo)
    contour_image_label.image = contour_photo
    contour_image_with_lines_label.config(image=contour_photo_with_lines)
    contour_image_with_lines_label.image = contour_photo_with_lines

def update_results_display(distances_with_lines, average_distance_with_lines, distances_in_microns_with_lines,
                           average_distance_microns_with_lines, diameter, area, fwhm_x, fwhm_y):
    # Clear the previous table entries
    for row in results_tree.get_children():
        results_tree.delete(row)
    
    # Insert new results into the table
    for i, distance in enumerate(distances_with_lines):
        results_tree.insert("", "end", values=(f"Distance with Lines {i+1} (pixels)", distance))
    
    results_tree.insert("", "end", values=("Average Distance with Lines (pixels)", f"{average_distance_with_lines:.2f}"))
    
    for i, distance in enumerate(distances_in_microns_with_lines):
        results_tree.insert("", "end", values=(f"Distance with Lines {i+1} (microns)", distance))
    
    results_tree.insert("", "end", values=("Average Distance with Lines (microns)", f"{average_distance_microns_with_lines:.2f}"))
    results_tree.insert("", "end", values=("Diameter (pixels)", diameter))
    results_tree.insert("", "end", values=("Area (pixels^2)", area))
    results_tree.insert("", "end", values=("FWHM X (pixels)", fwhm_x))
    results_tree.insert("", "end", values=("FWHM Y (pixels)", fwhm_y))

def create_scrollable_frame(container):
    canvas = Canvas(container)
    scrollbar = Scrollbar(container, orient='vertical', command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Bind mouse wheel event to canvas for global scrolling
    canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    return scrollable_frame

def save_table_as_csv():
    global uploaded_filename
    if uploaded_filename is None:
        print("No file uploaded")
        return

    file_path = os.path.splitext(uploaded_filename)[0] + ".csv"


    if file_path:
        with open(file_path, "w", newline='') as file:
            writer = csv.writer(file)
            for row in results_tree.get_children():
                row_values = results_tree.item(row)["values"]
                writer.writerow(row_values)
        print(f"Table saved as {file_path}")
        open_file_directory(file_path)

def open_file_directory(file_path):
    directory = os.path.dirname(file_path)
    os.startfile(directory)


def save_last_results(file_path, original_image, contour_image, contour_image_with_lines, distances_with_lines, 
                      average_distance_with_lines, distances_in_microns_with_lines, average_distance_microns_with_lines,
                      diameter, area, fwhm_x, fwhm_y):
    global scale_pixels_per_micron
    last_results = {
        "file_path": file_path,
        "original_image": original_image,
        "contour_image": contour_image,
        "contour_image_with_lines": contour_image_with_lines,
        "distances_with_lines": distances_with_lines,
        "average_distance_with_lines": average_distance_with_lines,
        "distances_in_microns_with_lines": distances_in_microns_with_lines,
        "average_distance_microns_with_lines": average_distance_microns_with_lines,
        "diameter": diameter,
        "area": area,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "scale_pixels_per_micron": scale_pixels_per_micron
    }
    with open("last_results.pkl", "wb") as f:
        pickle.dump(last_results, f)

def load_last_results():
    if not os.path.exists("last_results.pkl"):
        return None
    
    with open("last_results.pkl", "rb") as f:
        return pickle.load(f)

def calculate():
    global uploaded_filename, scale_pixels_per_micron
    if uploaded_filename is None:
        print("No file uploaded")
        return

    try:
        scale_pixels_per_micron = float(scale_entry.get())
    except ValueError:
        scale_pixels_per_micron = 10  # Default value if input is invalid

    try:
        (original_image, contour_image, contour_image_with_lines, distances_with_lines, 
        average_distance_with_lines, distances_in_microns_with_lines, average_distance_microns_with_lines,
        diameter, area, fwhm_x, fwhm_y) = process_image(uploaded_filename)
        
        update_image_display(original_image, contour_image, contour_image_with_lines)
        update_results_display(distances_with_lines, average_distance_with_lines, distances_in_microns_with_lines,
                               average_distance_microns_with_lines, diameter, area, fwhm_x, fwhm_y)
        save_last_results(uploaded_filename, original_image, contour_image, contour_image_with_lines, distances_with_lines, 
                          average_distance_with_lines, distances_in_microns_with_lines, average_distance_microns_with_lines,
                          diameter, area, fwhm_x, fwhm_y)
    except ValueError as e:
        print(f"Error during calculation: {e}")  # Debugging information

# Create the main window
root = Tk()
root.title("Image Analysis")
root.state('zoomed')  # Maximize the window to fit the screen

# Create frames for layout
top_frame = Frame(root)
top_frame.pack(side='top', fill='x')

content_frame = Frame(root)
content_frame.pack(side='top', fill='both', expand=True)

left_frame = Frame(content_frame)
left_frame.pack(side='left', fill='both', expand=True)

# Create scrollable frame for images and results
scrollable_frame = create_scrollable_frame(left_frame)

# Create a frame for the images
image_frame = Frame(scrollable_frame)
image_frame.pack(side='top', pady=10)

# Create a label for displaying the filename
filename_label = Label(image_frame, text="Filename: None")
filename_label.pack(side='top', pady=5)

# Create labels for displaying images
original_image_label = Label(image_frame)
original_image_label.pack(side='left', padx=10)

contour_image_label = Label(image_frame)
contour_image_label.pack(side='left', padx=10)

contour_image_with_lines_label = Label(image_frame)
contour_image_with_lines_label.pack(side='left', padx=10)

# Create treeview for displaying results in a table
results_tree = ttk.Treeview(scrollable_frame, columns=("Property", "Value"), show='headings', height=20)
results_tree.heading("Property", text="Property")
results_tree.heading("Value", text="Value")
results_tree.pack(side='top', pady=20)

# Make the table selectable
results_tree.tag_configure('selectable', font=('TkDefaultFont', 12))

# Set the width of the left column to expand
results_tree.column("Property", width=300, stretch='NO')
results_tree.column("Value", width=300, stretch='yes')

# Create button for opening file dialog
open_file_button = Button(top_frame, text="Open Image", command=open_file_dialog)
open_file_button.pack(side='left', padx=10, pady=10)

# Create button for exporting the table as a text file
export_button = Button(top_frame, text="Export Table as CSV", command=save_table_as_csv)
export_button.pack(side='left', padx=10, pady=10)

# Create entry for scale pixels per micron
scale_label = Label(top_frame, text="Scale (pixels per micron):")
scale_label.pack(side='left', padx=10, pady=10)

scale_entry = Entry(top_frame)
scale_entry.pack(side='left', padx=10, pady=10)

# Create button to calculate
calculate_button = Button(top_frame, text="Calculate", command=calculate)
calculate_button.pack(side='left', padx=10, pady=10)

# Load the last results if available
last_results = load_last_results()
if last_results:
    uploaded_filename = last_results["file_path"]
    update_filename_display(uploaded_filename)
    update_image_display(last_results["original_image"], last_results["contour_image"], last_results["contour_image_with_lines"])
    update_results_display(last_results["distances_with_lines"], last_results["average_distance_with_lines"], last_results["distances_in_microns_with_lines"],
                           last_results["average_distance_microns_with_lines"], last_results["diameter"], last_results["area"], last_results["fwhm_x"], last_results["fwhm_y"])
    scale_pixels_per_micron = last_results.get("scale_pixels_per_micron", 10)
    scale_entry.insert(0, str(scale_pixels_per_micron))

# Start the main loop
root.mainloop()
