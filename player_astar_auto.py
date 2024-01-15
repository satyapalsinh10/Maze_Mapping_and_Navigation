from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import re
import astar_4nodes as path_planner
import vanishing_points
import real_time_map
import shutil

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

def match_features(kp1, kp2, descriptors1, descriptors2):
        # Create a FLANN matcher object
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        # Keep good matches: ratio test as per Lowe's paper
        good_matches = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        similarity = len(good_matches) / len(kp1)

        return similarity

class KeyboardPlayerPyGame(Player):
    folder_path = 'images'
    target_image_path = 'given_target.png'

    
    def rank_images(self, folder_path, target_image_path, method='SIFT'):
        # Extract features from the target image
        target_keypoints, target_descriptors = extract_features(target_image_path, method)

        # Dictionary to hold images and their match counts
        image_match_counts = {}
        print("Starting SIFT similarity sequence!! ")
        # Loop over the images in the folder
        for image_name in os.listdir(folder_path):
            # Skip files that are not images (such as .DS_Store)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                continue

            image_path = os.path.join(folder_path, image_name)
            
            # Extract features from this image
            keypoints, descriptors = extract_features(image_path, method)
            
            # Match the features with the target
            good_matches = match_features(target_keypoints, keypoints, target_descriptors, descriptors)
            
            # Store the number of good matches
            image_match_counts[image_name] = good_matches
        
        # Rank the images by the number of good matches
        ranked_images = sorted(image_match_counts.items(), key=lambda item: item[1], reverse=True)
        #print(ranked_images)
        print("Done computing the descripter similarity and gola position is detected!!")
        return ranked_images

        # Path to your folder and target image
       

        # Run the ranking function
        #ranked_images = rank_images(folder_path, target_image_path, method='SIFT')  # or method='SURF'

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
            self.shortest_path = [(250, 250, 0), (251, 250, 0.0), (252, 250, 0.0), (253, 250, 0.0), (254, 250, 0.0), (255, 250, 0.0), (256, 250, 0.0), (257, 250, 0.0), (258, 250, 0.0), (259, 250, 0.0), (260, 250, 0.0), (261, 250, 0.0), (262, 250, 0.0), (263, 250, 0.0), (264, 250, 0.0), (265, 250, 0.0), (266, 250, 0.0), (267, 250, 0.0), (268, 250, 0.0), (269, 250, 0.0), (270, 250, 0.0), (271, 250, 0.0), (272, 250, 0.0), (273, 250, 0.0), (274, 250, 0.0), (275, 250, 0.0), (276, 250, 0.0), (277, 250, 0.0), (278, 250, 0.0), (279, 250, 0.0), (280, 250, 0.0), (281, 250, 0.0), (282, 250, 0.0), (283, 250, 0.0), (284, 250, 0.0), (285, 250, 0.0), (286, 250, 0.0), (287, 250, 0.0), (288, 250, 0.0), (289, 250, 0.0), (290, 250, 0.0), (291, 250, 0.0), (292, 250, 0.0), (293, 250, 0.0), (294, 250, 0.0), (295, 250, 0.0), (296, 250, 0.0), (297, 250, 0.0), (298, 250, 0.0), (299, 250, 0.0), (300, 250, 0.0), (301, 250, 0.0), (302, 250, 0.0), (303, 250, 0.0), (303, 251, 90.0), (303, 252, 90.0), (303, 253, 90.0), (303, 254, 90.0), (303, 255, 90.0), (303, 256, 90.0), (303, 257, 90.0), (303, 258, 90.0), (303, 259, 90.0), (303, 260, 90.0), (303, 261, 90.0), (303, 262, 90.0), (304, 262, 0.0), (305, 262, 0.0), (306, 262, 0.0), (307, 262, 0.0), (308, 262, 0.0), (309, 262, 0.0), (310, 262, 0.0), (311, 262, 0.0), (312, 262, 0.0), (313, 262, 0.0), (314, 262, 0.0), (315, 262, 0.0), (316, 262, 0.0), (317, 262, 0.0), (318, 262, 0.0), (318, 263, 90.0), (318, 264, 90.0), (318, 265, 90.0), (318, 266, 90.0), (318, 267, 90.0), (318, 268, 90.0), (318, 269, 90.0), (318, 270, 90.0), (318, 271, 90.0), (318, 272, 90.0), (318, 273, 90.0), (318, 274, 90.0), (318, 275, 90.0), (318, 276, 90.0), (317, 276, 180.0), (316, 276, 180.0), (315, 276, 180.0), (314, 276, 180.0), (313, 276, 180.0), (312, 276, 180.0), (311, 276, 180.0), (310, 276, 180.0), (309, 276, 180.0), (308, 276, 180.0), (307, 276, 180.0), (306, 276, 180.0), (305, 276, 180.0), (304, 276, 180.0), (303, 276, 180.0), (302, 276, 180.0), (301, 276, 180.0), (301, 277, 90.0), (301, 278, 90.0), (301, 279, 90.0), (301, 280, 90.0), (301, 281, 90.0), (301, 282, 90.0), (301, 283, 90.0), (301, 284, 90.0), (301, 285, 90.0), (301, 286, 90.0), (301, 287, 90.0), (301, 288, 90.0), (301, 289, 90.0), (301, 290, 90.0), (301, 291, 90.0), (301, 292, 90.0), (301, 293, 90.0), (301, 294, 90.0), (301, 295, 90.0), (301, 296, 90.0), (301, 297, 90.0), (301, 298, 90.0), (301, 299, 90.0), (301, 300, 90.0), (301, 301, 90.0), (301, 302, 90.0), (301, 303, 90.0), (301, 304, 90.0)]            
            self.path_index = 0
            self.curr_pnts = 0
            self.curr_waypoint = self.shortest_path[self.curr_pnts]
            self.short_flag = False
            self.v = vanishing_points.VanishingPointDetector()
            self.err = 0
            super().__init__()  # Now that self.map is defined, we can call the parent's __init__

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        
        self.orientation = 0
        # if self.phase != "pre-navigation":
        self.map.fill(0)
        self.position = (250, 250)
        self.curr_pnts = 0
        self.curr_waypoint = self.shortest_path[self.curr_pnts]
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
                new_position = (x + 1, y)
            elif self.orientation == 90:  # East
                new_position = (x, y + 1)
            elif self.orientation == 180:  # South
                new_position = (x - 1, y)
            else:  # West
                new_position = (x, y - 1)
        elif action == Action.BACKWARD:
            if self.orientation == 0:  # South
                new_position = (x - 1, y)
            elif self.orientation == 90:  # West
                new_position = (x, y - 1)
            elif self.orientation == 180:  # North
                new_position = (x + 1, y)
            else:  # East
                new_position = (x, y + 1)
        elif action == Action.LEFT:
            self.orientation = (self.orientation - 90) % 360
            return  # No need to check bounds or update position
        elif action == Action.RIGHT:
            self.orientation = (self.orientation + 90) % 360
            return  # No need to check bounds or update position

        # Check if the new position is within the bounds of the map and not a wall
        if 0 <= new_position[0] < self.map.shape[1] and 0 <= new_position[1] < self.map.shape[0]:
            self.position = new_position
        #print(self.position,self.orientation)

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
        desktop_path = "./"
        map_image_path = os.path.join(desktop_path, 'map_expo.png')
        pygame.image.save(map_surface, 'map_expo.png')
        # pygame.quit()

        coordinates_array = []  # Initialize an empty list to store coordinates

        # Iterate over the map and collect coordinates
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] == 1:  # Visited cell
                    coordinates = self.extract_coordinates(f'fpv_x{x}_y{y}.png')
                    if coordinates:
                        coordinates_array.append(coordinates)
        self.shortest_path = path_planner.calculate_path(coordinates_array, self.goal_position)
        coordinates_file_path = os.path.join(desktop_path, 'map_coordinates_expo.txt')
        with open(coordinates_file_path, 'w') as file:
            # Write all coordinates to the file
            
            file.write(f'array={coordinates_array}')

        return coordinates_array  # Return the array containing all coordinates
    
    def act(self):
        global i,turn
        action = Action.IDLE
        #print("short_flag = ", self.short_flag)
        #print(self.position, self.orientation)
        if not self.short_flag:
            #print("Inside first condiiton")
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
                                self.short_flag = True
                                self.position = (250, 250)
                                self.orientation = 0
                                self.curr_pnts = 0
                                self.curr_waypoint = self.shortest_path[self.curr_pnts]
                                
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
            #print("i", i)
            #import time
            if self.autonomous_movement_enabled:
                action = Action.FORWARD
                pygame.time.delay(50)  # Delay for 500 milliseconds (0.5 seconds)
                
            # Additional code for phase transition
            if self.phase == "pre-navigation":
                self.pre_navigation()

            self.update_position_and_orientation(action)
            self.update_map()
                    # Draw the map with the robot's new position                                
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_exiting = True
                    action = Action.QUIT
                
            if self.orientation == self.curr_waypoint[2] and turn == 0 : # this check is the proientaion criteria is met
                # Check if the key is in the keymap and set the corresponding action
                if self.position == (self.curr_waypoint[0],self.curr_waypoint[1]):
                    self.autonomous_movement_enabled = False # the waypoint reached
                    self.curr_pnts += 1
                    if self.curr_pnts > len(self.shortest_path)-3:
                        # If pre-navigation hasn't been done, start it
                        print("Reaced Goal Prof. CHEN FENG and Ashay !!!")
                        if not self.pre_navigation_done and self.phase == "exploration":
                            self.phase = "pre-navigation"
                            #self.pre_navigation()
                            #self.pre_navigation_done = True  # Set the flag to True
                        else:
                            # If pre-navigation is done, proceed to quit
                            self.is_exiting = True
                            self.short_flag = False
                            #action = Action.QUIT
                    self.curr_waypoint = self.shortest_path[self.curr_pnts]
                else:
                    #print("moving forward")
                    self.autonomous_movement_enabled = True  # the waypoint is yet to reached
                    
            elif i >= 37 or i == 0:
                    if self.curr_waypoint[2] == 0:
                        #print(self.orientation, self.curr_waypoint[2])
                        if self.orientation - self.curr_waypoint[2] < 0:
                                        # Introduce a delay for LEFT and RIGHT actions
                                        # Delay in milliseconds
                                #print("turning left, 0")
                                turn = -1
                                i = 0
                        elif self.orientation - self.curr_waypoint[2] > 0:
                                        # Introduce a delay for LEFT and RIGHT actions
                                        # Delay in milliseconds
                                #print("turning right, 0")
                                turn = +1
                                i = 0                     
                    else:
                        #print(self.position,self.orientation,self.curr_waypoint)
                        if self.orientation - self.curr_waypoint[2] > 0:
                                        # Introduce a delay for LEFT and RIGHT actions
                                        # Delay in milliseconds
                                #print("turning left")
                                turn = +1
                                i = 0
                        elif self.orientation - self.curr_waypoint[2] < 0:
                                        # Introduce a delay for LEFT and RIGHT actions
                                        # Delay in milliseconds
                                #print("turning right")
                                turn = -1
                                i = 0                
            update = True
            
            if turn==1:
                if i < 37:
                    action = Action.LEFT
                    i = i + 1
                else:
                    turn = 0
                    action = Action.IDLE
            elif turn==-1:
                if i < 37:
                    action = Action.RIGHT
                    i = i + 1
                else:
                    turn = 0
                    action = Action.IDLE
            else:
                #print("err = ", self.err)
                #print("before correction = ", self.position, self.orientation)

                if self.err > 5 and self.err < 15:
                    action = Action.RIGHT
                    print("Turning right for correction")
                    self.err = 0
                    update = False

                elif self.err < -5 and self.err > -15:
                    action = Action.LEFT
                    print("Turning left for correction")
                    self.err = 0
                    update = False

            # If autonomous movement is enabled, move forward automatically
            if self.autonomous_movement_enabled and update:
                action = Action.FORWARD
                #pygame.time.delay(100)  # Delay for 500 milliseconds (0.5 seconds)

            if update and ( i <= 0 or i >= 37):
                self.update_position_and_orientation(action)

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

        self.err = self.v.detect_vanishing_point(fpv)
        # if self.initiate:
        if self.get_state():
            _, phase, _, _, _, _ = self.get_state()
        self.save_fpv  = (self.save_fpv + 1) % 3

        # Save the FPV image according to the mapping
        if self.save_fpv  % 3 == 0 and not self.short_flag:
            self.save_fpv_image(fpv)
        self.initiate = True

        #live plot code commented for now
        #coordinates_array = []
        # Iterate over the map and collect coordinates
        #for y in range(self.map.shape[0]):
        #    for x in range(self.map.shape[1]):
        #        if self.map[y, x] == 1:  # Visited cell
        #            coordinates = self.extract_coordinates(f'fpv_x{x}_y{y}.png')
        #            if coordinates:
        #                coordinates_array.append(coordinates)
        #plotter = real_time_map.RealTimePlotter(coordinates_array)
        
        #plotter.plot_data()

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
            ranked_images = self.rank_images(self.folder_path, self.target_image_path, 'SIFT')

            if ranked_images:
                best_image_name, _ = ranked_images[0]
                goal_location = self.extract_coordinates(best_image_name)

                if goal_location:
                    self.goal_position = goal_location
                    print(f"The goal location is: {self.goal_position}")
                    # self.mark_goal_in_image5()  # Uncomment this if the method is defined
                    self.draw_map() 
                else:
                    print("No coordinates could be extracted from the image name.")
            else:
                print("No ranked images available to extract the goal location.")

            self.phase = "navigation"

if __name__ == "__main__":
    import vis_nav_game

    images_folder_path = 'images'
    
    # Ensure the 'images' folder exists
    if os.path.exists(images_folder_path):    
       shutil.rmtree(images_folder_path) 
       print('Image folder deleted and ready to go')

    player = KeyboardPlayerPyGame()
    vis_nav_game.play(the_player=player)