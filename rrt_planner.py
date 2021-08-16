#
# RRT Planner 0.0.0
#
# This node plans a path using RRT and
# RETURNS: it as a Path.msg whenever
# a GetPath.srv is triggered.
#

# See the ROS2 Built-In Messages repo for definitions
# from packages like "nav_msgs", "geometry_msgs", and more.
# https://github.com/ros2/common_interfaces
import rclpy
from rclpy.node import Node
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Pose, PoseStamped
from duckie_msgs.msg import Obstacle, ObstacleList # See duckie_msgs for definition

import copy
import math
import random
import numpy as np


class RRTNode:
    def __init__(self):
        """You must use this representation of a RRT Node when building up your tree.
        PARAMETERS:
            point: geometry_msgs/Point.msg object, a ROS2 point message to place the node in the map spatially.
            parent: `RRTNode', another instance of `RRTNode` that is the parent of this node. """
        self.point = Point()
        self.parent = None


class RRTPlanner(Node):
    def __init__(self):
        """You may add parameters & methods to this class/file as necessary. DON'T change existing methods or arguments of this class.
        PARAMETERS:
            plan_service: ROS2 Service that calls `plan_callback` when triggered.
            obstacles_subscriber: ROS2 Topic Subscriber subscribes to latest set obstacles in map. Here map is static and obstacles don't move. So set of obstacles is constant.
            obstacles: duckie_msgs/ObstacleList.msg object. Obstacle set returned by obstacles subscription. See duckie_msgs/ObstacleList.msg for message definition.
            root: `RRTNode`, root node of the RRT representing the starting point from which planning is started.
            rrt_tree: list[RRTNode] data structure to hold your RRT. """
        super().__init__('duckie_rrt_planner')
        self.plan_service = self.create_service(GetPlan, 'generate_rrt_plan', self.plan_callback)
        self.obstacles_subscriber = self.create_subscription(ObstacleList, '/obstacles', self.obstacles_listener, 10)
        self.obstacles = None
        self.root = RRTNode()
        self.rrt_tree = []
        self.radius = .15
        self.diameter = .3

    def obstacles_listener(self, msg):
        self.obstacles = msg

    def plan_callback(self, request, response):
        """ Plan a path using RRT algo.
         - GetPlanRequest/GetPlanResponse definition: https://github.com/ros2/common_interfaces
         - Implement and use `sampleFree`, `nearest`, `steer`, `obstacleFree`, `withinToleranceOfGoal` and `smoothPath` methods.
         - Don't exit RRT algo until solution is found (Ignore choosing N in this implementation but it's critical to assure probabilistic completeness). 
        PARAMETERS: Both GetPlanRequest objects. To use this message, see GetPlan service definition.
            request: Use "request.start.pose.position" as start node of graph, "request.goal.pose.position" as goal point, and "request.tolerance" as goal tolerance.
            response: nav_msgs/Path.msg
        RETURNS: response: Populate response.plan.poses array with geometry_msgs/PoseStamped.msgs intermediate waypoints to follow.
                 PoseStamped messages can be used for 3D planning, but only populate relevant fields (ignore header and orientation of PathStamped message). """
        newNode = self.root
        newNode.point = request.start.pose.position
        tree = self.rrt_tree
        tree.append(newNode)
        
        while True:
            if (self.withinToleranceOfGoal(newNode, request.goal.pose.position, request.tolerance)):
                goalNode = RRTNode()
                goalNode.point, goalNode.parent = request.goal.pose.position, newNode
                tree.append(goalNode)
                break
            random_point = self.randomSample(self.obstacles) # geometry_msgs/Point.msg object, randomly sampled configuration from free space.
            nearest_node = self.nearest(random_point) # `RRTNode` existing node in tree nearest to random_point
            newPoint = self.steer(nearest_node, random_point) # geometry_msgs/Point.msg object, point achievable by robot.
            if (self.obstacleFree(nearest_node, newPoint, self.obstacles)): # boolean True if segment is obstacle free, else False
                newNode = RRTNode()
                newNode.point, newNode.parent = newPoint, nearest_node
                tree.append(newNode)

        currNode, path = goalNode, []
        while (currNode.point != self.root.point):
            path.append(currNode)
            currNode = currNode.parent
        path.append(self.root)
        path.reverse()
        # smoothedPath = self.smoothPath(path, self.obstacles) # list[RRTNode], subset of nodes from given path

        finalPath = response.plan.poses
        for node in path: # Response has Path plan which has array poses of PoseStamped which has Pose pose which has Point position
            poseStamped = PoseStamped()
            poseStamped.pose.position = node.point
            finalPath.append(poseStamped)
        return response

    def randomSample(self, obstacles):
        """Uniform random sampling from the free configuration space.
        You are to build a obstacle configuration space from set of obstacles. Duckiebot is highly irregular in shape, so model it as a circle with radius 0.15 meters.
        PARAMETERS:
            obstacles: duckie_msgs/ObstacleList.msg object contains bounds of map & current positions of all obstacles. Note obstacles are rectangles with:
                - (x, y): bottom left corner of the rectangle
                - (width, height): size of the rectangle
        RETURNS: geometry_msgs/Point.msg object randomly sampled configuration from free space. Given in a plane, ignore the z dimension. """
        radius, randomX, randomY, pointFree = self.radius, None, None, False
        while not pointFree:
            pointFree = True
            randomX = radius + random.random() * (obstacles.map_width - self.diameter)
            randomY = radius + random.random() * (obstacles.map_height - self.diameter)
            for obstacle in obstacles.obs:
                bottom, top = obstacle.y - radius, obstacle.y + obstacle.height + radius
                left, right = obstacle.x - radius, obstacle.x + obstacle.width + radius
                if bottom <= randomY <= top and left <= randomX <= right:
                    pointFree = False
                    break
        randomPoint = Point()
        randomPoint.x, randomPoint.y = randomX, randomY
        return randomPoint

    def nearest(self, random_point):
        """Find the nearest vertex in tree to given point.
        PARAMETERS:
            random_point: geometry_msgs/Point.msg object point sampled from free configuration space.
        RETURNS:`RRTNode` existing node in tree nearest to random_point. """
        rrtTree, nearestNode, minDist = self.rrt_tree, None, None
        x = random_point.x
        y = random_point.y
        for node in rrtTree:
            xDiff = x - node.point.x
            yDiff = y - node.point.y
            distance = math.sqrt(xDiff * xDiff + yDiff * yDiff)
            if minDist is None or distance < minDist:
                nearestNode, minDist = node, distance
        return nearestNode

    def steer(self, nearest_node, random_point):
        """Find a point achievable by the robot that is between the nearest and goal points.
        Do NOT edit this method. In an advanced robotics planning course, you could use this
        `steer` function to account for the robot's pose, velocities, and dynamic capabilities
        at the `nearest_node`, thus creating a smooth plan. To simplify the task here, we will
        assume the robot is only executing simple translation and rotation in place plans.
        This means the plan will consist of straightline components that can be arbitrarily
        long, so this method simply RETURNS: the `random_point` directly.
        PARAMETERS:
            nearest_node: `RRTNode` node in RRT tree
            random_point: geometry_msgs/Point.msg object point sampled from free configuration space.
        RETURNS: geometry_msgs/Point.msg object, a point achievable by robot."""
        return random_point

    def obstacleFree(self, nearest_node, achievable_point, obstacles):
        """Check if segment between nearest and achievable falls in free configuration space.
        You are to build an obstacle configuration space from obstacle set. Duckiebot is highly irregular in shape, so model it as a circle with radius 0.15 meters.
        PARAMETERS:
            nearest_node: `RRTNode`node in RRT tree.
            achievable_point: geometry_msgs/Point.msg object achievable point in free configuration space.
            obstacles: duckie_msgs/ObstacleList.msg object contains bounds of map and current positions of obstacles which are rectangles with:
                - (x, y): bottom left corner of rectangle
                - (width, height): size of rectangle
        RETURNS: boolean True if segment obstacle free, else False. """

        def ccw(A,B,C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        A = nearest_node.point.x, nearest_node.point.y
        B = achievable_point.x, achievable_point.y
        r = self.radius
        for obstacle in obstacles.obs:
            left, right = obstacle.x - r, obstacle.x + obstacle.width + r
            bottom, top = obstacle.y - r, obstacle.y + obstacle.height + r
            leftEdge = (left, bottom), (left, top)
            topEdge = (left, top), (right, top)
            rightEdge = (right, bottom), (right, top)
            bottomEdge = (left, bottom), (right, bottom)
            obstacleEdges = [leftEdge, topEdge, rightEdge, bottomEdge]
            for edge in obstacleEdges:
                C, D = edge[0], edge[1]
                if (ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)):
                    return False
        return True

    def withinToleranceOfGoal(self, possible_node, goal_point, tolerance_m):
        """Check if a new possible_node is within tolerance_m of the goal_point.
        PARAMETERS:
            possible_node: `RRTNode` node in your RRT tree to check if close enough to goal.
            goal_point: geometry_msgs/Point.msg object goal point given in the GetPlanRequest object.
            tolerance_m: float distance from goal point considered close enough, from GetPlanRequest object.
        RETURNS: boolean True if possible_node is within tolerance_m of goal_point, else False. """
        dx, dy = goal_point.x - possible_node.point.x, goal_point.y - possible_node.point.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= tolerance_m:
            return True
        return False

    def smoothPath(self, path, obstacles): #LOOK INTO THIS
        """Iterate through the path eliminating nodes that are unnecessary
        RRT may plan paths that have more nodes than necessary. To eliminate
        a particular unnecessary node, x, simply check if the segment formed
        from nodes before, x-1, and after, x+1, is obstacle free. If so, you
        may eliminate node x.
        PARAMETERS:
        path: list[RRTNode] path represented by a list of `RRTNode` objects.
        obstacles: duckie_msgs/ObstacleList.msg object contains bounds of map & current obstacle positions which are rectangles with:
            - (x, y): bottom left corner of rectangle
            - (width, height): size of rectangle
        RETURNS: list[RRTNode], a subset of nodes from given path. """
        madeChange = True
        while madeChange:
            madeChange = False
            if len(path) > 2:
                index = 1
                while index < len(path)-1:
                    rrtNodePrev = path[index-1]
                    rrtNodeNext = path[index+1]
                    if (self.obstacleFree(rrtNodePrev, rrtNodeNext.point, obstacles)):
                        del path[index]
                        madeChange = True
                    else:
                        index += 1
        return path


def main():
    rclpy.init()
    duckie_rrt_planner = RRTPlanner()
    duckie_rrt_planner.get_logger().info("started!")

    try:
        rclpy.spin(duckie_rrt_planner)
    except KeyboardInterrupt:
        pass
    duckie_rrt_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()