import numpy as np
import cv2 as cv

import pyclipper
from obstacle_model import *

class environment:
    """Class contains all the funcitonality to create and manipulate environemnt and its visualization
    """

    
    def inflate_polygon(self, vertices, radius):
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(vertices, pyclipper.PT_CLIP, pyclipper.ET_CLOSEDPOLYGON)
        vertices_inflated = pco.Execute(radius)
        vertices_inflated = [tuple(x) for x in vertices_inflated[0]]
        return vertices_inflated

    def __init__(self, height, width, inflation_radius) -> None:
        """Initialize environment parameters

        Args:
            height (int): height of the map
            width (int): width of the map
        """
        self.height = height
        self.width = width
        self.inflation_radius = inflation_radius
        #create a map fo given dimentions. 3 channels for opencv BGR
        self.map = np.ones((height, width, 3))

        

        #create obstacle models for all the objects in the environment
        maze_points = [(250, 167), (251, 167), (252, 167), (253, 167), (254, 167), (255, 167), (256, 167), (257, 167), (258, 167), (259, 167), (260, 167), (261, 167), (262, 167), (263, 167), (264, 167), (265, 167), (266, 167), (267, 167), (268, 167), (269, 167), (270, 167), (271, 167), (272, 167), (273, 167), (274, 167), (275, 167), (276, 167), (277, 167), (278, 167), (279, 167), (280, 167), (281, 167), (282, 167), (283, 167), (284, 167), (285, 167), (286, 167), (287, 167), (288, 167), (289, 167), (290, 167), (291, 167), (292, 167), (293, 167), (294, 167), (295, 167), (296, 167), (297, 167), (298, 167), (299, 167), (300, 167), (301, 167), (302, 167), (303, 167), (304, 167), (305, 167), (306, 167), (307, 167), (308, 167), (309, 167), (310, 167), (311, 167), (312, 167), (313, 167), (314, 167), (315, 167), (316, 167), (317, 167), (318, 167), (319, 167), (250, 168), (319, 168), (250, 169), (319, 169), (250, 170), (319, 170), (250, 171), (319, 171), (250, 172), (319, 172), (250, 173), (319, 173), (250, 174), (319, 174), (250, 175), (319, 175), (250, 176), (319, 176), (250, 177), (319, 177), (250, 178), (319, 178), (250, 179), (319, 179), (250, 180), (263, 180), (264, 180), (265, 180), (266, 180), (267, 180), (268, 180), (269, 180), (270, 180), (271, 180), (272, 180), (273, 180), (274, 180), (275, 180), (276, 180), (277, 180), (319, 180), (320, 180), (321, 180), (322, 180), (323, 180), (324, 180), (325, 180), (326, 180), (327, 180), (328, 180), (329, 180), (330, 180), (331, 180), (250, 181), (263, 181), (277, 181), (331, 181), (263, 182), (277, 182), (331, 182), (263, 183), (277, 183), (331, 183), (263, 184), (277, 184), (331, 184), (263, 185), (277, 185), (331, 185), (263, 186), (277, 186), (331, 186), (263, 187), (277, 187), (331, 187), (263, 188), (277, 188), (331, 188), (263, 189), (277, 189), (331, 189), (263, 190), (277, 190), (331, 190), (263, 191), (277, 191), (331, 191), (263, 192), (277, 192), (331, 192), (263, 193), (277, 193), (331, 193), (263, 194), (277, 194), (331, 194), (263, 195), (277, 195), (331, 195), (250, 196), (251, 196), (252, 196), (253, 196), (254, 196), (255, 196), (256, 196), (257, 196), (258, 196), (259, 196), (260, 196), (261, 196), (262, 196), (263, 196), (277, 196), (278, 196), (279, 196), (280, 196), (281, 196), (282, 196), (283, 196), (284, 196), (285, 196), (286, 196), (287, 196), (288, 196), (289, 196), (290, 196), (291, 196), (292, 196), (293, 196), (294, 196), (295, 196), (296, 196), (297, 196), (298, 196), (299, 196), (300, 196), (301, 196), (302, 196), (303, 196), (304, 196), (305, 196), (331, 196), (250, 197), (305, 197), (331, 197), (250, 198), (305, 198), (331, 198), (250, 199), (305, 199), (331, 199), (250, 200), (305, 200), (331, 200), (250, 201), (305, 201), (331, 201), (250, 202), (305, 202), (331, 202), (250, 203), (305, 203), (331, 203), (250, 204), (305, 204), (331, 204), (250, 205), (305, 205), (331, 205), (250, 206), (305, 206), (331, 206), (250, 207), (305, 207), (331, 207), (250, 208), (305, 208), (331, 208), (250, 209), (305, 209), (306, 209), (307, 209), (308, 209), (309, 209), (310, 209), (311, 209), (312, 209), (313, 209), (314, 209), (315, 209), (316, 209), (317, 209), (318, 209), (319, 209), (331, 209), (250, 210), (319, 210), (331, 210), (250, 211), (319, 211), (331, 211), (250, 212), (319, 212), (331, 212), (250, 213), (319, 213), (331, 213), (250, 214), (319, 214), (331, 214), (250, 215), (319, 215), (331, 215), (250, 216), (319, 216), (331, 216), (250, 217), (319, 217), (331, 217), (250, 218), (319, 218), (331, 218), (250, 219), (319, 219), (331, 219), (250, 220), (319, 220), (331, 220), (250, 221), (319, 221), (331, 221), (250, 222), (319, 222), (331, 222), (250, 223), (319, 223), (331, 223), (250, 224), (304, 224), (305, 224), (306, 224), (307, 224), (308, 224), (309, 224), (310, 224), (311, 224), (312, 224), (313, 224), (314, 224), (315, 224), (316, 224), (317, 224), (318, 224), (319, 224), (331, 224), (250, 225), (304, 225), (331, 225), (250, 226), (304, 226), (331, 226), (250, 227), (304, 227), (331, 227), (250, 228), (304, 228), (331, 228), (250, 229), (304, 229), (331, 229), (250, 230), (304, 230), (331, 230), (250, 231), (304, 231), (331, 231), (250, 232), (304, 232), (331, 232), (250, 233), (304, 233), (331, 233), (250, 234), (304, 234), (331, 234), (250, 235), (304, 235), (331, 235), (250, 236), (304, 236), (331, 236), (250, 237), (304, 237), (305, 237), (306, 237), (307, 237), (308, 237), (309, 237), (310, 237), (311, 237), (312, 237), (313, 237), (314, 237), (315, 237), (316, 237), (317, 237), (318, 237), (319, 237), (320, 237), (321, 237), (322, 237), (323, 237), (324, 237), (325, 237), (326, 237), (327, 237), (328, 237), (329, 237), (330, 237), (331, 237), (250, 238), (250, 239), (250, 240), (250, 241), (250, 242), (250, 243), (250, 244), (250, 245), (250, 246), (250, 247), (250, 248), (250, 249), (250, 250)]#create original boundary obstacle model
        self.boundary_model = obstacle_model([maze_points])
        
        #create inflated boundary obstacle model
        self.inflated_boundary_model = obstacle_model([])
        
        #create original polygon objects obstacle model
        self.original_obstacle_model = obstacle_model([])
        #self.original_obstacle_model = obstacle_model([
        #    [(150,0), (150,100), (100,100), (100,0)],                           # Polygon corresponding to botom rectangular pillar
        #    [(150,150), (150,250), (100,250), (100,150)],                       # Polygon corresponding to Top rectangular pillar
        #    [(360,87), (360,163), (300,200), (240,163), (240,87), (300,50)],    # Polygon corresponding to hexgon 
        #    [(510,125), (460,225), (460,25) ],                                  # Polygon corresponding to Triangle 
        #])      

        self.inflated_obstacle_model = obstacle_model([])
        #create inflated polygon objects obstacle model
        #self.inflated_obstacle_model = obstacle_model([
        #    self.inflate_polygon([(150,0), (150,100), (100,100), (100,0)],  self.inflation_radius),
        #    self.inflate_polygon([(150,150), (150,250), (100,250), (100,150)],  self.inflation_radius),
        #    self.inflate_polygon([(360,87), (360,163), (300,200), (240,163), (240,87), (300,50)],  self.inflation_radius),
        #    self.inflate_polygon([(510,125), (460,225), (460,25) ],  self.inflation_radius),  
        #])  


    
    def create_map(self):
        """Idnetify obstacles and free space in the map uisng the obstacle models
        """
        #Iterate through all the states in the enviornement
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                #Checks if state present inside the non-inflated obstacle
                if self.original_obstacle_model.is_inside_obstacle((j,i)):
                    self.map[i,j] = [0x80, 0x80, 0xff]
                
                #Checks if state present inside an nflated obstacle
                elif self.inflated_obstacle_model.is_inside_obstacle((j,i)):
                    self.map[i,j] = [0x7a, 0x70, 0x52]

                #Checks if state present outside the inflated boundary
                elif not self.inflated_boundary_model.is_inside_obstacle((j,i)):
                    self.map[i,j] = [0x7a, 0x70, 0x52]

                #Identify as stae belongs to the free space
                else:
                    self.map[i,j] = [0xc2, 0xba, 0xa3]

    def is_valid_position(self, position):
        """Checks if a given position belongs to free space or obstacle space

        Args:
            position (tuple): state of the agent to be verified

        Returns:
            bool: True if pixel belongs to free space else False
        """
        print(position)
        if position[0] >=0 and position[0] < 520 and position[1] >=0 and position[1] < 520:
            print(position)
            #Check if a state belongs to free psace by comparing the color of the respective pixel in the environment
            if  self.map[position[0], position[1]][0] == 0xc2 and \
                self.map[position[0], position[1]][1] == 0xba and \
                self.map[position[0], position[1]][2] == 0xa3:
                return True
        
        #return false of state is in obstacle space
        return False


    def refresh_map(self):
        """Refreshes map with updated image map
        """
        #Flip the map to satisfy the environment direction
        image = cv.flip(self.map.astype('uint8'), 0)
        image = cv.resize(image, (0,0), fx = 2, fy = 2)
        cv.imshow("map", image)

    def update_map(self, position):
        """Highlights a point in the environment at the given locaiton

        Args:
            position (tuple): pixel location
        """
        i, j, _ = position
        self.map[i, j] = [255, 255, 255]
        self.refresh_map()

    def update_action(self, start, end):
        """Update map with explored nodes colour

        Args:
            explored_node (tuple): state that has been visited
        """
        x1, y1, theta1 = start
        x2, y2, theta2 = end
        self.map = cv.arrowedLine(self.map, (y1, x1), (y2, x2),[0, 255, 255], 1, cv.LINE_AA)
        self.refresh_map()


    def save_image(self, file_path):
        """saves current state of the environment in the file location as image 

        Args:
            file_path (string): absolute path of the file where the image needs to be saved
        """
        #Flip the map to satisfy the environment direction
        image = cv.flip(self.map.astype('uint8'), 0)
        image = cv.resize(image, (0,0), fx = 2, fy = 2)
        cv.imwrite(file_path, image)

    def highlight_state(self, position, size, color):
        """Draws a circle at the given location in the environment

        Args:
            position (tuple): pixel location
        """
        self.map =  cv.circle(self.map, (position[1],position[0]), size, color, -1)
        self.refresh_map()

    def highlight_point(self, position):
        """Highlights a point in the environment at the given locaiton

        Args:
            position (tuple): pixel location
        """
        i, j, _ = position
        self.map[i, j] = [255, 0, 0]

    def show_robot(self, position, size):
        """Highlights a point in the environment at the given locaiton

        Args:
            position (tuple): pixel location
        """
        _map = self.map.copy()
        image =  cv.circle(_map, (position[1],position[0]), size, (255, 0, 0), -1)
        image = cv.flip(image.astype('uint8'), 0)
        resized = cv.resize(image, (0,0), fx = 2, fy = 2)
        cv.imshow("map", resized)
        return image

    #primitives to save video of the jplanning environment-----------------------
    def begin_video_writer(self):
         self.writer= cv.VideoWriter('Animation Video.mp4', cv.VideoWriter_fourcc(*'DIVX'), 10, (self.width, self.height))
    

    def insert_video_frame(self, image):
        self.writer.write(image)

    def write_video_frame(self):
        image = cv.flip(self.map.astype('uint8'), 0)
        # image = cv.resize(image, (0,0), fx = 2, fy = 2)
        self.writer.write(image)

    def close_video_writer(self):
        self.writer.release()
    #--------------------------------------------------------------------------------
    

if __name__ == "__main__":
    _map_viz = environment(250, 600, 10)
    _map_viz.create_map()
    _map_viz.refresh_map()
    cv.waitKey(0)
