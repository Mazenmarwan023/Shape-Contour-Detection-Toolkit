from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog,
                             QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor
import numpy as np
import cv2
import math


class Contour(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Object Analyzer")
        self.setGeometry(200, 200, 1500, 1200)
        self.main_window = main_window
        self.image = None
        self.processed_image = None
        self.roi_center = None
        self.roi_radius = 80  # Default radius
        self.alpha = 5.0  # Internal energy weight (smoothness)
        self.beta = 10.0  # External energy weight (edge attraction)
        self.gamma = 3.0  # Balloon energy weight (expansion/contraction)
        self.numOfIterations = 50  # Number of iterations for contour refinement
        self.snake_points = None
        self.contour_area = 0
        self.contour_perimeter = 0
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QGridLayout()
        controls_layout = QVBoxLayout()

        group_box = QGroupBox()
        box_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        buttons_layout = QHBoxLayout()

        # Image labels
        input_image_layout = QVBoxLayout()
        self.input_label = QLabel("Original Image")
        self.input_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFixedSize(500, 500)
        self.color_mode = QRadioButton("Color")
        self.gray_mode = QRadioButton("Grayscale")
        self.color_mode.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.color_mode)
        mode_layout.addWidget(self.gray_mode)
        input_image_layout.addWidget(self.input_label)
        input_image_layout.addLayout(mode_layout)

        self.output_label = QLabel("Processed Image")
        self.output_label.setStyleSheet("background-color: black; border: 1px solid black;")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedSize(500, 500)

        images_layout.addLayout(input_image_layout)
        images_layout.addWidget(self.output_label)

        self.upload_button = QPushButton("Upload Image")
        self.reset_button = QPushButton("Reset")
        self.save_button = QPushButton("Save")
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.reset_button)

        # Next pages buttons
        next_pages_buttons_layout=QHBoxLayout()
        self.home_page_button=QPushButton("Home page")
        self.home_page_button.clicked.connect(self.switch_to_home_page)
        self.home_page_button.setFixedWidth(150)
        next_pages_buttons_layout.addWidget(self.home_page_button)
        next_pages_buttons_layout.addStretch(1)

        box_layout.addLayout(next_pages_buttons_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)

        # Active Contour Parameters
        self.radius_input = QLineEdit(str(self.roi_radius))
        self.radius_input.setPlaceholderText("ROI Radius")
        self.alpha_input = QLineEdit(str(self.alpha))
        self.alpha_input.setPlaceholderText("Alpha (Curvature)")
        self.beta_input = QLineEdit(str(self.beta))
        self.beta_input.setPlaceholderText("Beta (Edge Adherence)")
        self.gamma_input = QLineEdit(str(self.gamma))
        self.gamma_input.setPlaceholderText("Gamma (Balloon Effect)")
        self.iterations_input = QLineEdit(str(self.numOfIterations))
        self.iterations_input.setPlaceholderText("Iterations")

        # Results display
        self.area_label = QLabel("Area: 0")
        self.perimeter_label = QLabel("Perimeter: 0")
        self.area_label.setStyleSheet("font-weight: bold;")
        self.perimeter_label.setStyleSheet("font-weight: bold;")

        controls_layout.addWidget(QLabel("Contour Parameters:"))
        controls_layout.addWidget(self.radius_input)
        controls_layout.addWidget(self.alpha_input)
        controls_layout.addWidget(self.beta_input)
        controls_layout.addWidget(self.gamma_input)
        controls_layout.addWidget(self.iterations_input)
        controls_layout.addWidget(QLabel("Results:"))
        controls_layout.addWidget(self.area_label)
        controls_layout.addWidget(self.perimeter_label)

        self.analyze_button = QPushButton("Analyze Contour")
        self.analyze_button.clicked.connect(self.contourUpdating)
        controls_layout.addWidget(self.analyze_button)

        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)

        main_layout.addLayout(controls_layout, 0, 0)
        main_layout.addWidget(group_box, 0, 1)
        main_layout.setColumnStretch(1, 2)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", options=options
        )
        if file_path:
            self.image = cv2.imread(file_path)
            self.processed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.image, self.input_label)
            self.roi_center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
            self.initialize_snake()

    def display_image(self, img, label):
        if len(img.shape) == 2:  # Grayscale image
            q_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        else:  # Color image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def reset_images(self):
        self.input_label.clear()
        self.output_label.clear()
        self.image = None
        self.processed_image = None
        self.snake_points = None
        self.contour_area = 0
        self.contour_perimeter = 0
        self.area_label.setText("Area: 0")
        self.perimeter_label.setText("Perimeter: 0")

    def save_output_image(self):
        """Save the processed image."""
        if self.processed_image is None:
            print("Error: No image to save.")
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            print(f"Processed image saved to {file_path}.")

    def initialize_snake(self):
        """Initialize snake points around the ROI circle"""
        if self.roi_center is None or self.image is None:
            return

        try:
            self.roi_radius = int(self.radius_input.text())
        except:
            self.roi_radius = 80

        n_points = 100  # Number of points in the snake
        self.snake_points = []
        for i in range(n_points):
            angle = 2 * math.pi * i / n_points
            x = self.roi_center[0] + self.roi_radius * math.cos(angle)
            y = self.roi_center[1] + self.roi_radius * math.sin(angle)
            self.snake_points.append([x, y])

    def calculate_gradient_magnitude(self, image):
        """Calculate gradient magnitude of the image (edge strength) with proper normalization"""
        grad_x = np.zeros(image.shape, dtype=np.float32)
        grad_y = np.zeros(image.shape, dtype=np.float32)
        
        # Calculate x gradient (central difference)
        for y in range(1, image.shape[0]-1):
            for x in range(1, image.shape[1]-1):
                grad_x[y, x] = image[y, x+1] - image[y, x-1]
        
        # Calculate y gradient (central difference)
        for y in range(1, image.shape[0]-1):
            for x in range(1, image.shape[1]-1):
                grad_y[y, x] = image[y+1, x] - image[y-1, x]
        
        # Gradient magnitude with normalization
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        if grad_mag.max() > 0:
            grad_mag = (grad_mag / grad_mag.max()) * 255
        return grad_mag

    def calculate_contour_area(self, points):
        """Calculate area of a polygon using the shoelace formula"""
        if len(points) < 3:
            return 0
        
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0

    def calculate_contour_perimeter(self, points):
        """Calculate perimeter of a polygon by summing edge lengths"""
        if len(points) < 2:
            return 0
        
        perimeter = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            perimeter += math.sqrt(dx*dx + dy*dy)
        return perimeter

    def is_point_in_roi_circle(self, point):
        """Check if a point is within the ROI circle"""
        if self.roi_center is None:
            return False
        distance = math.sqrt((point[0] - self.roi_center[0])**2 + (point[1] - self.roi_center[1])**2)
        return distance < self.roi_radius

    def constrain_point_to_circle(self, point):
        """Move a point to the nearest position within the ROI circle"""
        if self.roi_center is None:
            return point
        
        cx, cy = self.roi_center
        px, py = point
        dx = px - cx
        dy = py - cy
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance <= self.roi_radius:
            return point
        
        # Project point onto circle boundary
        scale = self.roi_radius / distance
        return (cx + dx * scale, cy + dy * scale)

    def contourUpdating(self):
        """Active contour algorithm with points constrained to ROI circle"""
        if self.image is None or self.processed_image is None:
            return

        # Get parameters from UI with validation
        
        self.alpha = max(0.1, float(self.alpha_input.text()))
        self.beta = max(0.1, float(self.beta_input.text()))
        self.gamma = max(0.1, float(self.gamma_input.text()))
        self.numOfIterations = max(10, min(500, int(self.iterations_input.text())))
        

        # Initialize snake if not already done
        if self.snake_points is None:
            self.initialize_snake()

        # Convert to numpy array for easier manipulation
        snake = np.array(self.snake_points, dtype=np.float32)
        n_points = len(snake)

        # Calculate normalized gradient magnitude (edge map)
        grad_mag = self.calculate_gradient_magnitude(self.processed_image)
        
        # Pre-calculate average point distance for continuity term
        if len(snake) > 1:
            avg_dist = np.mean(np.sqrt(np.sum((np.roll(snake, -1, axis=0) - snake)**2, axis=1)))
        else:
            avg_dist = 1.0

        for iteration in range(self.numOfIterations):
            new_snake = snake.copy()

            for i in range(n_points):
                x, y = snake[i]
                prev_point = snake[(i-1) % n_points]
                next_point = snake[(i+1) % n_points]

                # Calculate internal energy (smoothness)
                # Continuity term (encourage even spacing)
                cont_term = self.alpha * (
                    ((x - prev_point[0])**2 + (y - prev_point[1])**2 - avg_dist**2) +
                    ((next_point[0] - x)**2 + (next_point[1] - y)**2 - avg_dist**2)
                )

                # Curvature term (discourage sharp bends)
                curvature = self.beta * (
                    (prev_point[0] - 2*x + next_point[0])**2 +
                    (prev_point[1] - 2*y + next_point[1])**2
                )

                # Get neighborhood (5x5 window for better search)
                min_x = max(0, int(x) - 2)
                max_x = min(self.processed_image.shape[1]-1, int(x) + 2)
                min_y = max(0, int(y) - 2)
                max_y = min(self.processed_image.shape[0]-1, int(y) + 2)

                best_energy = float('inf')
                best_point = [x, y]

                # Search in neighborhood for best position
                for nx in range(min_x, max_x + 1):
                    for ny in range(min_y, max_y + 1):
                        if nx == x and ny == y:
                            continue

                        # Check if point is within ROI circle
                        if not self.is_point_in_roi_circle((nx, ny)):
                            continue

                        # External energy (edge attraction + balloon force)
                        edge_strength = grad_mag[ny, nx]
                        
                        # Balloon force (expand or contract)
                        # Calculate normal direction
                        dx = next_point[0] - prev_point[0]
                        dy = next_point[1] - prev_point[1]
                        nx_dir = -dy  # Normal x component
                        ny_dir = dx   # Normal y component
                        norm = max(1, math.sqrt(nx_dir**2 + ny_dir**2))
                        nx_dir /= norm
                        ny_dir /= norm
                        
                        # Dot product with displacement vector
                        disp_x = nx - x
                        disp_y = ny - y
                        balloon_force = (disp_x * nx_dir + disp_y * ny_dir) * self.gamma

                        total_energy = cont_term + curvature - edge_strength - balloon_force

                        if total_energy < best_energy:
                            best_energy = total_energy
                            best_point = [nx, ny]

                # If no valid point found in neighborhood, constrain to circle
                if best_energy == float('inf'):
                    best_point = self.constrain_point_to_circle([x, y])

                # Update snake point
                new_snake[i] = best_point

            # Update snake with relaxation (0.5 is the relaxation factor)
            snake += 0.5 * (new_snake - snake)

            # Ensure all points are within the ROI circle after update
            for i in range(n_points):
                if not self.is_point_in_roi_circle(snake[i]):
                    snake[i] = self.constrain_point_to_circle(snake[i])

        # Update the snake points
        self.snake_points = snake.tolist()

        # Calculate area and perimeter
        self.contour_area = self.calculate_contour_area(self.snake_points)
        self.contour_perimeter = self.calculate_contour_perimeter(self.snake_points)
        
        # Update the UI with the results
        self.area_label.setText(f"Area: {self.contour_area:.2f} px²")
        self.perimeter_label.setText(f"Perimeter: {self.contour_perimeter:.2f} px")

        # Draw the results
        self.draw_results()

    def draw_results(self):
        """Improved drawing with better visualization"""
        if self.image is None:
            return

        # Create a copy of the original image
        result_img = self.image.copy()

        # Draw the initial ROI circle (dark green)
        center = (int(self.roi_center[0]), int(self.roi_center[1]))
        radius = int(self.roi_radius)
        cv2.circle(result_img, center, radius, (0, 100, 0), 2)

        # Draw the active contour (blue)
        if self.snake_points:
            snake = np.array([[int(x), int(y)] for (x, y) in self.snake_points], dtype=np.int32)
            
            # Draw lines between points
            for i in range(len(snake)):
                x1, y1 = snake[i]
                x2, y2 = snake[(i+1) % len(snake)]
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw points for better visibility
            for (x, y) in snake:
                cv2.circle(result_img, (x, y), 2, (0, 0, 255), -1)

        # Display the result
        self.display_image(result_img, self.output_label)
        self.processed_image = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    
    def switch_to_home_page(self):
        self.main_window.stacked_widget.setCurrentIndex(0)