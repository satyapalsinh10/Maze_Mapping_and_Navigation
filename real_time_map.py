import cv2
import numpy as np

class RealTimePlotter:
    def __init__(self, map_data):
        self.map_data = map_data
        self.window_size = (800, 600)
        self.scale_factor = 20  # Adjust the scale factor as needed
        self.coordinates_array = []

    def plot_data(self):
        frame = np.ones((self.window_size[1], self.window_size[0], 3), dtype=np.uint8) * 255  # White background

        for y in self.map_data:
            coordinates = y
            print(coordinates)
            if coordinates:
                scaled_coordinates = (
                    int(coordinates[0] * self.scale_factor),
                    int(coordinates[1] * self.scale_factor)
                )
                cv2.circle(frame, scaled_coordinates, 5, (255, 0, 0), -1)

        cv2.imshow('Real-time Plotting', frame)
        key = cv2.waitKey(1000)

if __name__ == "__main__":
    # Replace this with your actual map data
    map_data = np.random.randint(2, size=(20, 20, 2))

    plotter = RealTimePlotter(map_data)
    plotter.plot_data()
