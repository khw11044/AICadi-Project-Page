import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_msg = Twist()
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s
        self.get_logger().info("Robot Control Node Initialized")

    def control_robot(self, command: str):
        """키 입력에 따라 로봇을 제어하는 함수"""
        if command == 'w':  # 전진
            self.cmd_msg.linear.x = self.linear_speed
            self.cmd_msg.angular.z = 0.0
        elif command == 's':  # 후진
            self.cmd_msg.linear.x = -self.linear_speed
            self.cmd_msg.angular.z = 0.0
        elif command == 'a':  # 좌회전
            self.cmd_msg.linear.x = 0.0
            self.cmd_msg.angular.z = self.angular_speed
        elif command == 'd':  # 우회전
            self.cmd_msg.linear.x = 0.0
            self.cmd_msg.angular.z = -self.angular_speed
        else:  # 정지
            self.cmd_msg.linear.x = 0.0
            self.cmd_msg.angular.z = 0.0
        
        # 메시지 퍼블리시
        self.publisher.publish(self.cmd_msg)
        self.get_logger().info(f"Command executed: {command}")

def init_ros_control():
    """ROS2 노드 초기화 (중복 방지)"""
    if not rclpy.utilities.ok():  # 이미 초기화되었는지 확인
        rclpy.init()
    return RobotControlNode()
