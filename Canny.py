from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox,QComboBox, 
                             QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
from collections import defaultdict
from scipy.signal import convolve2d  

from contour import Contour


class Canny(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canny")
        self.setGeometry(200,200,1500,1200)
        self.image = None  # To store the loaded image
        self.equalized_image = None  # To store the loaded image

        self.contour_page= Contour(self)


        self.initUi()

    def initUi(self):
        self.main_widget = QWidget()

        main_layout = QGridLayout()
        self.stacked_widget= QStackedWidget()
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.contour_page)
        self.stacked_widget.setCurrentWidget(self.main_widget)
        controls_layout_left = QVBoxLayout()
        controls_layout_right=QVBoxLayout()


        group_box = QGroupBox()
        box_layout=QVBoxLayout()
        images_layout=QHBoxLayout()
        buttons_layout=QHBoxLayout()

        # Labels for images
        input_image_layout=QVBoxLayout()
        self.input_label = QLabel("Original Image")
        self.input_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFixedSize(500, 500)
        self.color_mode = QRadioButton("Color")
        self.gray_mode = QRadioButton("Grayscale")
        self.color_mode.setChecked(True)  # Default mode is Color
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.color_mode)
        mode_layout.addWidget(self.gray_mode)
        input_image_layout.addWidget(self.input_label)
        input_image_layout.addLayout(mode_layout)

        self.output_label = QLabel("Noisy Image")
        self.output_label.setStyleSheet("background-color: black; border: 1px solid black;")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedSize(500, 500)

       
        images_layout.addLayout(input_image_layout)
        images_layout.addWidget(self.output_label)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setFixedWidth(150)
        self.reset_button=QPushButton("Reset")
        self.reset_button.setFixedWidth(150)
        self.save_button=QPushButton("Save")
        self.save_button.setFixedWidth(150)
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.reset_button)

        # Next pages buttons
        next_pages_buttons_layout=QHBoxLayout()
        self.contour_button=QPushButton("Contour")
        self.contour_button.setFixedWidth(150)
        next_pages_buttons_layout.addWidget(self.contour_button)
 

        box_layout.addLayout(next_pages_buttons_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)


        # Canny edge
        canny_edge_label=QLabel("Canny Edge")
        self.low_threshold = QSlider()
        self.high_threshold = QSlider()
        self.apply_canny=QPushButton("Apply Canny")

        # Hough transform
        hough_transform=QLabel("Hough Transform")
        self.combobox= QComboBox()
        self.combobox.addItems(["Line","Circle","Ellipse"])

        self.line_slider=QSlider(Qt.Horizontal)
        self.line_slider.setMinimum(100)
        self.line_slider.setMaximum(200)

        self.line_slider = QSlider()


        circle_transform=QLabel("Circle transform")
        self.circle_min_radius = QSlider()
        self.circle_max_radius = QSlider()
        self.circle_threshold = QSlider()


        ellipse_transform=QLabel("Ellipse transform")
        self.ellipse_min_radius = QSlider()
        self.ellipse_max_radius = QSlider()
        self.ellipse_threshold = QSlider()



        self.apply_transform=QPushButton("Apply")

        controls_layout_left.addWidget(canny_edge_label)
        controls_layout_left.addStretch(1)
        controls_layout_left.addWidget(QLabel("Low threshold"))
        self.add_slider_with_label(controls_layout_left, self.low_threshold, 1, 100)
        controls_layout_left.addWidget(QLabel("Hight threshold"))
        self.add_slider_with_label(controls_layout_left, self.high_threshold, 1, 255)
        controls_layout_left.addStretch(1)
        controls_layout_left.addWidget(self.apply_canny)


        controls_layout_right.addWidget(hough_transform)
        controls_layout_right.addWidget(self.combobox)
        controls_layout_right.addStretch(1)
        controls_layout_right.addWidget(QLabel("Line threshold"))
        self.add_slider_with_label(controls_layout_right, self.line_slider,  100, 200)
        controls_layout_right.addStretch(1)
        controls_layout_right.addWidget(circle_transform)
        controls_layout_right.addWidget(QLabel("Min radius"))
        self.add_slider_with_label(controls_layout_right, self.circle_min_radius, 10, 100)
        controls_layout_right.addWidget(QLabel("Max radius"))
        self.add_slider_with_label(controls_layout_right, self.circle_max_radius, 1, 150)
        controls_layout_right.addWidget(QLabel("Circle threshold"))
        self.add_slider_with_label(controls_layout_right, self.circle_threshold, 5, 200)
        controls_layout_right.addStretch(1)
        controls_layout_right.addWidget(ellipse_transform)
        controls_layout_right.addWidget(QLabel("Min radius"))
        self.add_slider_with_label(controls_layout_right, self.ellipse_min_radius, 0, 120)
        controls_layout_right.addWidget(QLabel("Max radius"))
        self.add_slider_with_label(controls_layout_right, self.ellipse_max_radius, 0, 120)
        controls_layout_right.addWidget(QLabel("Ellipse threshold"))
        self.add_slider_with_label(controls_layout_right, self.ellipse_threshold, 0, 120)
        controls_layout_right.addStretch(1)
        controls_layout_right.addWidget(self.apply_transform)

        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.contour_button.clicked.connect(self.switch_to_contour)
        self.apply_canny.clicked.connect(self.canny_edge_detection)
        self.apply_transform.clicked.connect(self.apply_hough_transform)


        main_layout.addLayout(controls_layout_left,0,0)
        main_layout.addWidget(group_box,0,1)
        main_layout.addLayout(controls_layout_right,0,2)
        # main_layout.setColumnStretch(1,2)


        self.setStyleSheet("""
             QLabel{
                font-size:20px;
                color:white;     
                    }
            QLabel#menu{
                font-size:29px;
                color:white;
                           }
            QPushButton{
                    font-size:18px;
                    padding:10px;
                    border:white 1px solid;
                    border-radius:15px;
                    background-color:white;
                    color:black;         
                        }

        """)
        
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.stacked_widget)

    # Create a function to add labeled sliders
    def add_slider_with_label(self,layout, slider, min_value, max_value):
        # Create labels
        value_label = QLabel(str(min_value))  # Label for current value

        # Set slider properties
        slider.setOrientation(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.valueChanged.connect(lambda value: value_label.setText(str(value)))  # Update value label

        # Create horizontal layout for slider + labels
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)

        # Add layout to main controls layout
        layout.addLayout(slider_layout)


    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", options=options
        )

        if file_path:
            # Check which mode is selected
            if self.gray_mode.isChecked():
                self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            else:
                self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Load as color (default)

            self.display_image(self.image, self.input_label)  # Display in input label

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def reset_images(self):
        self.input_label.clear()  # Clear input image label
        self.output_label.clear()  # Clear output image label
        self.image = None  # Remove stored image
        self.equalized_image = None  # Remove stored output

    def save_output_image(self):
        if self.equalized_image is None:
            return  # No image to save
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options
        )
        
        if file_path:
            cv2.imwrite(file_path, self.equalized_image)  # Save using OpenCV


    def gaussian_kernel(self,size, sigma):
        """Generates a 1D Gaussian kernel."""
        k = size // 2
        x = np.arange(-k, k + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        return kernel / kernel.sum()  # Normalize
    

    def apply_gaussian_filter(self,image, size=5, sigma=1.4):
        """Applies Gaussian blur manually using 1D separable convolution."""
        kernel = self.gaussian_kernel(size, sigma)

        # Apply 1D Gaussian filter in horizontal direction
        smoothed = convolve2d(image, kernel[:, None], mode="same", boundary="symm")

        # Apply 1D Gaussian filter in vertical direction
        smoothed = convolve2d(smoothed, kernel[None, :], mode="same", boundary="symm")
        
        return smoothed.astype(np.uint8)  # Convert to 8-bit

    def apply_sobel(self,image):
        """Applies the Sobel edge detection manually."""
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Compute gradients using convolution
        Gx = convolve2d(image, sobel_x, mode="same", boundary="symm")
        Gy = convolve2d(image, sobel_y, mode="same", boundary="symm")

        # Compute gradient magnitude
        magnitude = np.sqrt(Gx**2 + Gy**2)
        magnitude = (magnitude / magnitude.max()) * 255  # Normalize to [0, 255]
        
        # Compute gradient direction
        direction = np.arctan2(Gy, Gx)
        
        return magnitude.astype(np.uint8), direction



    def non_maximum_suppression(self, magnitude, direction):
        if len(magnitude.shape) == 3:  # Check for RGB image
            M, N, _ = magnitude.shape
        else:
            M, N = magnitude.shape

        output = np.zeros((M, N), dtype=np.float32)
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180  # Convert negative angles to positive range

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = r = 255  # Default values

                theta = angle[i, j]  # Ensure this is a scalar, not an array

                # Apply correct angle conditions
                if (0 <= theta < 22.5) or (157.5 <= theta <= 180):  # 0
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif (22.5 <= theta < 67.5):  # 45
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif (67.5 <= theta < 112.5):  # 90
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif (112.5 <= theta < 157.5): # 135
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # Non-maximum suppression logic
                output[i, j] = magnitude[i, j] if (magnitude[i, j] >= q) and (magnitude[i, j] >= r) else 0

        return output


    def double_thresholding(self, image, low_threshold, high_threshold):
        strong = 255
        weak = 75
        
        # Normalize the image to 0-255 if necessary
        image = np.uint8(255 * (image / np.max(image))) if np.max(image) > 0 else image

        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

        output = np.zeros_like(image, dtype=np.uint8)
        output[strong_i, strong_j] = strong
        output[weak_i, weak_j] = weak
        
        return output, strong, weak


    def edge_tracking(self, image, strong=255, weak=75):
        M, N = image.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if image[i, j] == weak:
                    # Instead of setting directly to 0, check connectivity first
                    neighbors = image[i-1:i+2, j-1:j+2].flatten()
                    if strong in neighbors:
                        image[i, j] = strong  # Convert to strong edge
                    else:
                        image[i, j] = 0  # Suppress unconnected weak edges

        return image



    def canny_edge_detection(self):
        if self.image is None:
            print("No image found.")
            return

        # Ensure grayscale conversion
        gray_image = self.image.copy()
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        low_threshold = self.low_threshold.value()
        high_threshold = self.high_threshold.value()

        smoothed = self.apply_gaussian_filter(gray_image)
        magnitude, direction = self.apply_sobel(smoothed)
        suppressed = self.non_maximum_suppression(magnitude, direction)
        thresholded, strong, weak = self.double_thresholding(suppressed, low_threshold, high_threshold)
        final_edges = self.edge_tracking(thresholded, strong, weak)

        self.equalized_image = final_edges
        self.display_image(self.equalized_image, self.output_label)
        print("doneee")


    def apply_hough_transform(self):
        selected_transform = self.combobox.currentText()  # Read selected transform from combo box

        if selected_transform == "Line":
            self.hough_line_transform()
        elif selected_transform == "Circle":
            self.hough_circle_transform()
        elif selected_transform == "Ellipse":
            self.hough_ellipse_transform()


    
    def hough_line_transform(self):
        if self.equalized_image is None:
            return  # Ensure an edge-detected image is available

        # Get the threshold value from the slider
        line_threshold = self.line_slider.value()

        # Get image dimensions
        height, width = self.equalized_image.shape

        # Define the maximum possible value for rho (image diagonal)
        diagonal = int(np.sqrt(height ** 2 + width ** 2))

        # Define rho and theta ranges
        rhos = np.arange(-diagonal, diagonal, 1)  # Step size of 1 for rho
        thetas = np.deg2rad(np.arange(-90, 90, 1))  # Convert degrees to radians

        # Create the accumulator array (votes)
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

        # Get edge points from the Canny output
        edge_points = np.argwhere(self.equalized_image > 0)

        # Precompute cos(theta) and sin(theta) values
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        # Voting process (optimized)
        for y, x in edge_points:  # For each edge pixel
            rhos_calc = (x * cos_thetas + y * sin_thetas).astype(int)  # Compute rho values for all thetas at once
            rho_indices = np.clip(rhos_calc + diagonal, 0, len(rhos) - 1)  # Map rho to index
            accumulator[rho_indices, np.arange(len(thetas))] += 1  # Increment votes in one operation

        # Extract lines based on threshold
        detected_lines = np.argwhere(accumulator > line_threshold)

        if len(self.image.shape) == 2:  # Ensure single-channel before conversion
            processed_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            processed_image = self.image

        # Draw detected lines
        for rho_idx, theta_idx in detected_lines:
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]

            # Convert (rho, theta) to two points for line drawing
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color lines

        # Display the output image with detected lines
        self.display_image(processed_image, self.output_label)

    def hough_circle_transform(self):
        if self.equalized_image is None:
            return  

        # Get threshold and radius range from sliders
        circle_threshold = self.circle_threshold.value()
        min_radius = self.circle_min_radius.value()
        max_radius = self.circle_max_radius.value()

        # Get image dimensions
        height, width = self.equalized_image.shape

        # Define radius range and create a 3D accumulator (y, x, radius)
        radius_range = np.arange(min_radius, max_radius + 1, 1)
        accumulator = np.zeros((height, width, len(radius_range)), dtype=np.int32)

        # Get edge points
        edge_points = np.argwhere(self.equalized_image > 0)

        # Precompute sin and cos values for angles (reduce step for performance)
        angles = np.deg2rad(np.arange(0, 360, 3))  # Step reduced from 1° to 3°  
        cos_thetas = np.cos(angles)
        sin_thetas = np.sin(angles)

        # Voting process (optimized with NumPy)
        for r_idx, r in enumerate(radius_range):
            a_vals = (edge_points[:, 1][:, None] - r * cos_thetas).astype(int)  # X-centers
            b_vals = (edge_points[:, 0][:, None] - r * sin_thetas).astype(int)  # Y-centers

            # Keep only valid (a, b) values within the image
            valid_indices = (a_vals >= 0) & (a_vals < width) & (b_vals >= 0) & (b_vals < height)
            a_vals, b_vals = a_vals[valid_indices], b_vals[valid_indices]

            # Increment votes in the accumulator
            np.add.at(accumulator[:, :, r_idx], (b_vals, a_vals), 1)

        # Find circles that exceed the threshold
        detected_circles = np.argwhere(accumulator > circle_threshold)

        # Convert grayscale image to BGR for visualization
        processed_image = self.image.copy()

        # Draw detected circles
        for y, x, r_idx in detected_circles:
            radius = radius_range[r_idx]
            cv2.circle(processed_image, (x, y), radius, (0, 255, 0), 2)  # Green color circles

        # Display output
        self.display_image(processed_image, self.output_label)




    def hough_ellipse_transform(self):
        if self.equalized_image is None:
            return  

        # Get threshold and radius range from sliders
        ellipse_threshold = self.ellipse_threshold.value()
        min_radius = self.ellipse_min_radius.value()
        max_radius = self.ellipse_max_radius.value()

        # Get image dimensions
        height, width = self.equalized_image.shape

        # Define axis ranges
        major_axis_range = np.arange(min_radius, max_radius + 1, 2)
        minor_axis_range = np.arange(int(min_radius * 0.5), max_radius + 1, 2)

        # Initialize accumulator (y, x, major_axis, minor_axis)
        accumulator = np.zeros((height, width, len(major_axis_range), len(minor_axis_range)), dtype=np.int32)

        # Get edge points
        edge_points = np.argwhere(self.equalized_image > 0)

        # Precompute sin and cos values for angles
        angles = np.deg2rad(np.arange(0, 360, 5))  # Increased step for efficiency
        cos_thetas = np.cos(angles)
        sin_thetas = np.sin(angles)

        # Voting process
        for a_idx, a in enumerate(major_axis_range):
            for b_idx, b in enumerate(minor_axis_range):
                x_centers = (edge_points[:, 1][:, None] - a * cos_thetas).astype(int)
                y_centers = (edge_points[:, 0][:, None] - b * sin_thetas).astype(int)

                # Keep only valid (x_center, y_center) within bounds
                valid_indices = (x_centers >= 0) & (x_centers < width) & (y_centers >= 0) & (y_centers < height)
                x_centers, y_centers = x_centers[valid_indices], y_centers[valid_indices]

                # Increment votes in the accumulator
                np.add.at(accumulator[:, :, a_idx, b_idx], (y_centers, x_centers), 1)

        # Extract potential ellipses that exceed the threshold
        detected_ellipses = np.argwhere(accumulator > ellipse_threshold)

        # Convert grayscale image to BGR for visualization
        processed_image = self.image.copy()
        
        ellipse_candidates = []
        for y, x, a_idx, b_idx in detected_ellipses:
            major_axis = major_axis_range[a_idx]
            minor_axis = minor_axis_range[b_idx]
            votes = accumulator[y, x, a_idx, b_idx]
            ellipse_candidates.append((x, y, major_axis, minor_axis, votes))

        # Sort ellipses by votes (highest first)
        ellipse_candidates.sort(key=lambda e: e[4], reverse=True)

        # Draw only the top detected ellipses
        for x, y, a, b, _ in ellipse_candidates:
            cv2.ellipse(processed_image, (x, y), (a, b), 0, 0, 360, (0, 255, 0), 2)

        # Display output
        self.display_image(processed_image, self.output_label)



    def switch_to_contour(self):
        self.stacked_widget.setCurrentIndex(1)



 










































