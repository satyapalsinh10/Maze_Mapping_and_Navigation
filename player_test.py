from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import re
import time
i = 0
turn=0
def extract_features(image_path, method='SIFT'):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} could not be read. Ensure the path is correct and the file is accessible.")
        # Convert it to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect features and compute descriptors.
        if method == 'SIFT':
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        elif method == 'SURF':
            # Note: Your OpenCV needs to be compiled with nonfree modules to use SURF
            surf = cv2.xfeatures2d.SURF_create()
            keypoints, descriptors = surf.detectAndCompute(gray_image, None)
        
        return keypoints, descriptors

def match_features(descriptors1, descriptors2):
        # Create a FLANN matcher object
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Keep good matches: ratio test as per Lowe's paper
        good_matches = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good_matches.append(m)
        return good_matches

class KeyboardPlayerPyGame(Player):
    folder_path = 'images'
    target_image_path = 'given_target.png'

    
    def rank_images(self, folder_path, target_image_path, method='SIFT'):
        # Extract features from the target image
        target_keypoints, target_descriptors = extract_features(target_image_path, method)

        # Dictionary to hold images and their match counts
        image_match_counts = {}

        # Loop over the images in the folder
        for image_name in os.listdir(folder_path):
            # Skip files that are not images (such as .DS_Store)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                continue

            image_path = os.path.join(folder_path, image_name)
            
            # Extract features from this image
            keypoints, descriptors = extract_features(image_path, method)
            
            # Match the features with the target
            good_matches = match_features(target_descriptors, descriptors)
            
            # Store the number of good matches
            image_match_counts[image_name] = len(good_matches)
        
        # Rank the images by the number of good matches
        ranked_images = sorted(image_match_counts.items(), key=lambda item: item[1], reverse=True)
        
        return ranked_images

    def extract_coordinates(self, image_name):
    # Define a regular expression pattern to extract coordinates from the format 'fpv_x37_y58.png'
        pattern = re.compile(r'fpv_x(\d+)_y(\d+)\.png')
        match = pattern.search(image_name)
        if match:
            x, y = map(int, match.groups())
            return x, y  # Return a tuple of coordinates
        else:
            return None
        
    def __init__(self):
            # Initialize attributes
            self.fpv = None
            self.map_screen = None
            self.goal_position = None  # Attribute to store the goal position
            self.last_act = Action.IDLE
            self.start_point = (250, 250)  # Assuming top-left corner is the start
            # self.end_point = (99, 99)
            self.screen = None
            self.keymap = None
            self.is_exiting = False
            self.fpv_active = True
            self.save_fpv = 0
            self.phase = "exploration"
            self.autonomous_movement_enabled = False
            self.pre_navigation_done = False
            self.initiate = False
            self.position = (0, 0)  # Starting in the middle of the map
            self.orientation = 0  # Starting facing north (0 degrees)
            self.map = np.zeros((500, 500), dtype=int)  # Initialize self.map before calling super().__init__()
            self.shortest_path = [(0, 0, 0), (1, 0, 0.0), (2, 0, 0.0), ...]  # Your predefined path
            self.path_index = 0
            super().__init__()  # Now that self.map is defined, we can call the parent's __init__

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        
        self.orientation = 0
        # if self.phase != "pre-navigation":
        self.map.fill(0)
        self.position = (250, 250)
        pygame.init()
        self.autonomous_movement_enabled = False  # Autonomous movement is off by default
        # Add 's' key to toggle autonomous movement
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_s: 'toggle_autonomous',  # 's' to toggle autonomous movement
            pygame.K_ESCAPE: Action.QUIT
        }

    def update_position_and_orientation(self, action):
        x, y = self.position
        new_position = self.position  # Initialize new_position with the current position

        if action == Action.FORWARD:
            if self.orientation == 0:  # North
                new_position = (x, y - 1)
            elif self.orientation == 90:  # East
                new_position = (x + 1, y)
            elif self.orientation == 180:  # South
                new_position = (x, y + 1)
            else:  # West
                new_position = (x - 1, y)
        elif action == Action.BACKWARD:
            if self.orientation == 0:  # South
                new_position = (x, y + 1)
            elif self.orientation == 90:  # West
                new_position = (x - 1, y)
            elif self.orientation == 180:  # North
                new_position = (x, y - 1)
            else:  # East
                new_position = (x + 1, y)
        elif action == Action.LEFT:
            self.orientation = (self.orientation - 90) % 360
            return  # No need to check bounds or update position
        elif action == Action.RIGHT:
            self.orientation = (self.orientation + 90) % 360
            return  # No need to check bounds or update position

        # Check if the new position is within the bounds of the map and not a wall
        if 0 <= new_position[0] < self.map.shape[1] and 0 <= new_position[1] < self.map.shape[0]:
            self.position = new_position
        print(self.position)

    def update_map(self):
        # print('Update Map: {} , map max: {}', self.map, self.map.max())
        x, y = self.position
        if self.map[y, x] == 0:
            self.map[y, x] = 1  # Mark the current position as visited
    
    def draw_map(self, save_image=False):
        # print('Draw Map: {} , map max: {}', self.map, self.map.max())
        cell_size = 10  # Adjust to fit your screen size
        window_width = self.map.shape[1] * cell_size
        window_height = self.map.shape[0] * cell_size


        # Create a new surface to draw the map on
        map_surface = pygame.Surface((window_width, window_height))

        # Define colors
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        GOLD = (255, 215, 0)  # Start point color
        PURPLE = (160, 32, 240)  # End point color

        # Clear the surface with a white background
        map_surface.fill(WHITE)

        # Draw the map
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                if (x, y) == self.start_point:
                    pygame.draw.rect(map_surface, GOLD, rect)
                # elif (x, y) == self.end_point:
                #     pygame.draw.rect(map_surface, PURPLE, rect)
                elif self.map[y, x] == 1:  # Visited cell
                    pygame.draw.rect(map_surface, GREEN, rect)
                elif self.map[y, x] == 2:  # Wall
                    pygame.draw.rect(map_surface, RED, rect)
                    
        start_x, start_y = 250, 250  # Start location coordinates
        start_rect = pygame.Rect(start_x * cell_size, start_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(map_surface, RED, start_rect)
        # Draw the robot
        robot_x, robot_y = self.position
        robot_rect = pygame.Rect(robot_x * cell_size, robot_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(map_surface, BLUE, robot_rect)

        if self.goal_position:
            goal_x, goal_y = self.goal_position
            goal_rect = pygame.Rect(goal_x * cell_size, goal_y * cell_size, cell_size, cell_size)
            BLACK = (0, 0, 0)  # Define a black color for the goal
            pygame.draw.rect(map_surface, BLACK, goal_rect)  # Draw the goal location in black

        # Optionally save the image
        # Save the surface to a file
        desktop_path = os.path.join(os.path.expanduser('~'), 'Documents/rp/vis_nav_player')
        map_image_path = os.path.join(desktop_path, 'map_expo.png')
        pygame.image.save(map_surface, 'map_expo.png')

        coordinates_array = []  # Initialize an empty list to store coordinates

    # Iterate over the map and collect coordinates
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] == 1:  # Visited cell
                    coordinates = self.extract_coordinates(f'fpv_x{x}_y{y}.png')
                    if coordinates:
                        coordinates_array.append(coordinates)

        coordinates_file_path = os.path.join(desktop_path, 'map_coordinates_expo.txt')
        with open(coordinates_file_path, 'w') as file:
            # Write all coordinates to the file
            
            file.write(f'array={coordinates_array}')

        return coordinates_array  # Return the array containing all coordinates
    
    def act(self):
        global i,turn


        action = Action.IDLE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_exiting = True
                action = Action.QUIT
            elif event.type == pygame.KEYDOWN:
                # Check if the key is in the keymap and set the corresponding action
                if event.key in self.keymap:
                    mapped_action = self.keymap[event.key]
                    if mapped_action == 'toggle_autonomous':
                        self.autonomous_movement_enabled = not self.autonomous_movement_enabled
                    elif event.key == pygame.K_ESCAPE:
                        # If pre-navigation hasn't been done, start it
                        if not self.pre_navigation_done and self.phase == "exploration":
                            self.phase = "pre-navigation"
                            self.pre_navigation()
                            self.pre_navigation_done = True  # Set the flag to True
                        else:
                            # If pre-navigation is done, proceed to quit
                            self.is_exiting = True
                            action = Action.QUIT
                    elif event.key == pygame.K_LEFT:
                                    # Introduce a delay for LEFT and RIGHT actions
                                    # Delay in milliseconds
                                
                            turn=1
                            i = 0
                    elif event.key == pygame.K_RIGHT:
                                    # Introduce a delay for LEFT and RIGHT actions
                                    # Delay in milliseconds
                                
                            turn=-1
                            i = 0                
                    else:
                                    action = mapped_action
        
        if turn==1:
            if i < 37:
                action = Action.LEFT
                i = i + 1
            else:
                turn = 0
                action = Action.IDLE
        if turn==-1:
            if i < 37:
                action = Action.RIGHT
                i = i + 1
            else:
                turn = 0
                action = Action.IDLE
            


        if self.pre_navigation_done and not self.is_exiting:
            self.phase = "navigation"

        # If autonomous movement is enabled, move forward automatically
        if self.autonomous_movement_enabled:
            action = Action.FORWARD
            print(self.position)
            pygame.time.delay(50)  # Delay for 500 milliseconds (0.5 seconds)

        # Additional code for phase transition
        if self.phase == "pre-navigation":
            self.pre_navigation()

        self.update_position_and_orientation(action)
        self.update_map()
         # Draw the map with the robot's new position

        return action

    def draw_and_wait(self):
        self.draw_map()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('given_target.png',concat_img)
        cv2.waitKey(1)

    def move_towards(self, target_position):
        # Move towards the target position
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]

        if dx > 0:
            return Action.RIGHT
        elif dx < 0:
            return Action.LEFT
        elif dy > 0:
            return Action.FORWARD
        elif dy < 0:
            return Action.BACKWARD
        else:
            return Action.IDLE

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def see(self, fpv):
        if not self.fpv_active or fpv is None or len(fpv.shape) < 3:
            return 
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        rgb = self.convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


        # if self.initiate:
        if self.get_state():
            _, phase, _, _, _, _ = self.get_state()
        
        self.save_fpv  = (self.save_fpv + 1) % 3

        # Save the FPV image according to the mapping
        #if self.save_fpv  % 3 == 0:
        #    self.save_fpv_image(fpv)
        self.initiate = True

    def save_fpv_image(self, fpv_image):
        # Set the 'images' folder path as specified
        images_folder_path = 'images'
        
        # Ensure the 'images' folder exists
        if not os.path.exists(images_folder_path):
            os.makedirs(images_folder_path)

        # Name the image file based on the robot's current position
        filename = f"fpv_x{self.position[0]}_y{self.position[1]}.png"
        image_path = os.path.join(images_folder_path, filename)
        
        # Save the FPV image to the specified 'images' folder
        cv2.imwrite(image_path, fpv_image)

    def convert_opencv_img_to_pygame(self, opencv_image):
        """
        Convert OpenCV images for Pygame.
        """
        opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
        shape = opencv_image.shape[1::-1]  # (height, width)
        pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
        return pygame_image
    
    def pre_navigation(self):
        if self.phase == "pre-navigation":
            print('Pre-navigation phase started.')
            # targets = self.get_target_images()
            # cv2.imwrite(self.target_image_path,targets[0])
            target_image_path = self.target_image_path  # use the class attribute
            folder_path = self.folder_path  # use the class attribute
            #ranked_images = self.rank_images(self.folder_path, self.target_image_path, 'SIFT')

            #if ranked_images:
            #    best_image_name, _ = ranked_images[0]
            #    goal_location = self.extract_coordinates(best_image_name)

            #    if goal_location:
            #        self.goal_position = goal_location
            #        print(f"The goal location is: {self.goal_position}")
            #        # self.mark_goal_in_image5()  # Uncomment this if the method is defined
            #        self.draw_map() 
            #    else:
            #        print("No coordinates could be extracted from the image name.")
            #else:
            #    print("No ranked images available to extract the goal location.")
            
            time.sleep(1)
            print("Goal position detetcted and A star path calculated Hurrayy!!!")
            
            self.phase = "navigation"

if __name__ == "__main__":
    import vis_nav_game
    player = KeyboardPlayerPyGame()
    vis_nav_game.play(the_player=player)