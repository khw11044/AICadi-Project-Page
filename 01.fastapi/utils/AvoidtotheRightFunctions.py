import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time
class ObstacleAvoidanceNode(Node):
    
    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.get_logger().info('Obstacle Avoidance Node Initialized')
        # 라이다 데이터 구독
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        # 속도 명령 퍼블리셔
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        # 속도 명령 메시지
        self.cmd_msg = Twist()
        # 장애물 탐지 범위 설정
        self.obstacle_distance_threshold = 0.5  # cm
        self.min_intensity_threshold = 1000.0  # 반사 강도가 너무 낮은 값 필터링

        #회전 복구를 위한 카운터
        self.angularZ_count = 0
        self.linearX_count = 0

        # 초기값 설정
        self.ranges = []
        self.intensities = []  # 반사 강도 데이터
        self.range_min = 0.0
        self.range_max = 0.0

        #가장 가까운 거리
        self.range_front_left = 0.0 #instance variable
        self.range_front_right = 0.0 #instance variable
        self.Closest = 0.0
        self.closest_distance_left = 0

    def scan_callback(self, msg):
        self.ranges = msg.ranges  # 라이다 거리 데이터
        self.intensities = msg.intensities  # 라이다 반사 강도 데이터
        self.range_min = msg.range_min  # 최소 유효 거리
        self.range_max = msg.range_max  # 최대 유효 거리

        #거리 함수들
        self.range_front_left = self.Function_front_left()
        self.Closest = self.Function_CLOSEST()
    
    # def Function_front_left(self):
    #     # filtered_ranges = self.filter_noise(self.ranges, self.intensities, self.range_min, self.range_max)
    #     front_left_ranges = self.ranges[79:300] # +,- 
    #     front_left_distances = [r for r in front_left_ranges if self.range_min <= r <= self.range_max]
    #     # self.get_logger().info(f"Filtered Front Left Distances: {front_left_distances}")  # 필터링 후 확인

    #     self.range_front_right = min(front_left_distances) if front_left_distances else float('inf')
    #     # print(f"closeset_distance_front {self.range_front_left}")
    #     return self.range_front_right
    
    def Function_CLOSEST(self):
        
        filtered_ranges = self.filter_noise(self.ranges, self.intensities, self.range_min, self.range_max)
        self.Closest = min(filtered_ranges) if filtered_ranges else float('inf')
        return self.Closest
    
    def Function_front_left(self):
        # filtered_ranges = self.filter_noise(self.ranges, self.intensities, self.range_min, self.range_max)
        front_left_ranges = self.ranges[79:300] # +,- 
        front_left_distances = [r for r in front_left_ranges if self.range_min <= r <= self.range_max]
        # self.get_logger().info(f"Filtered Front Left Distances: {front_left_distances}")  # 필터링 후 확인

        self.range_front_left = min(front_left_distances) if front_left_distances else float('inf')
        # print(f"closeset_distance_front {self.range_front_left}")
        return self.range_front_left
    
    # def Control(self, ranges, intensities, range_min, range_max):
        # filtered_ranges = self.filter_noise(msg.ranges, msg.intensities, msg.range_min, msg.range_max)

        # front_left_ranges = ranges[76:300] # +,- 
        # front_left_distances = [r for r in front_left_ranges if range_min <= r <= range_max]
        # self.range_front_left = min(front_left_distances) if front_left_distances else float('inf')

        # left_range = ranges[75]
        # closest_distance = self.scan_callback(msg)
        # right_ranges = ranges[401:601]
        # left_ranges = ranges[0:149]

                # 유효한 거리 값만 필터링
        # filtered_ranges = self.filter_noise(ranges, intensities, range_min, range_max)
        # for i in range(len(msg.ranges)):
        #     if 290<= i <= 310:
        #         print(msg.ranges[i])

        # 가장 가까운 장애물 거리 확인
        # THE_CLOSEST = min(filtered_ranges) if filtered_ranges else float('inf')
        # 가장 가까운 거리 확인

        # print("Front_Left:", self.Function_front_left())
        # if THE_CLOSEST <= self.obstacle_distance_threshold: #라이더 센서에 장애물이 감지 됐을때
            # if left_range <= self.obstacle_distance_threshold:
            #     self.AngleRestoration()
            
        #     if self.range_front_left < self.obstacle_distance_threshold:
        #         self.avoid_obstacle()
        # else:
        #     self.move_forward()
        # print(f"장애물과의 거리: {self.range_front_left}")
        # if self.range_front_left(ranges, intensities, range_min, range_max) < self.obstacle_distance_threshold:
        #     self.avoidToTheRight()
        # else:
        #     self.move_forward()

    def AngleRestoration(self):
        self.angularZ_count -= 1
        # for i in range(self.angularZ_count):
        self.cmd_msg.angular.z = 0.3
        self.cmd_msg.angular.x = 0.0
        self.publisher.publish(self.cmd_msg)
        # print("angle restoring!")
        print("angle restoring! amount angle left:", self.angularZ_count)
        time.sleep(0.05)
        self.publisher.publish(self.cmd_msg)

    def AvoidToTheRight(self):
        """
        충돌 회피 동작.
        - 후진 또는 회전 명령을 발행.
        """
        # self.get_logger().info("Obstacle detected! Avoiding...")
        self.cmd_msg.linear.x = 0.0  # 뒤로 이동
        self.cmd_msg.angular.z = -0.3  # 회전 - 오른쪽 + 왼쪽
        self.angularZ_count += 1
        print(f"회피 오른쪽 회전 횟수: {self.angularZ_count}")
        self.publisher.publish(self.cmd_msg)

    def filter_noise(self, ranges, intensities, range_min, range_max):
        """
        라이다 데이터의 노이즈를 필터링.
        - 최소/최대 거리 필터링
        - 반사 강도(intensity) 기반 필터링
        """
        filtered = []
        for r, i in zip(ranges, intensities):
            if range_min <= r <= range_max and i >= self.min_intensity_threshold:
                filtered.append(r)
        return filtered
    

    def move_forward(self):
        """
        장애물이 없을 때 앞으로 이동.
        """
        self.cmd_msg.linear.x = 0.1  # 직진
        self.cmd_msg.angular.z = 0.0
        self.publisher.publish(self.cmd_msg)
    
    def stop(self):
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = 0.0
        self.publisher.publish(self.cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()