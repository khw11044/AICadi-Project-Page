import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import Twist
from AvoidtotheRightFunctions import ObstacleAvoidanceNode
# args=None
def main():
        rclpy.init()
        buffer_distance = 0.2
        right_cnt = 0
        left_cnt = 0
        #노드 생성
        node = ObstacleAvoidanceNode()
                # 라이다 데이터가 수신될 때까지 대기
        while not node.ranges:  # ranges가 빈 상태인지 확인
                rclpy.spin_once(node, timeout_sec=0.1)

        print("LaserScan data received.")
        avoid_start = False
        # 장애물 거리 출력
        try:
                while rclpy.ok():
                        rclpy.spin_once(node, timeout_sec=0.1)  # 콜백 처리
                        
                        if node.range_front_left >= node.obstacle_distance_threshold + buffer_distance:
                                if node.angularZ_count > 0 :
                                        print('복구중 / angularZ_count:', node.angularZ_count)
                                        node.AngleRestoration()
                                        left_cnt += 1
                                        
                                # elif node.angularZ_count == 0 and right_cnt != 0:
                                #         node.AvoidToTheLeft()
                                #         right_cnt -= 1
                                #         print(f"복구 왼쪽 회전 횟수: {right_cnt}")
                                # node.move_forward()
                        
                        if node.range_front_left < node.obstacle_distance_threshold:
                                if not avoid_start:
                                        avoid_start = True 
                                        right_cnt = 0
                                        print('우측 회피를 시작합니다.')
                                print('우측 회피 중 / angularZ_count:', node.angularZ_count)
                                node.AvoidToTheRight()
                                right_cnt += 1
                                print('right_cnt ->', right_cnt)
                        else:
                                print('앞으로 가기.')
                                node.move_forward()
                                
                                if node.angularZ_count == 0 and right_cnt != 0:
                                        node.AvoidToTheLeft()
                                        right_cnt -= 1
                                        print(f"복구 왼쪽 회전 횟수: {right_cnt}")
                                
                        if avoid_start and node.angularZ_count == 0 and right_cnt ==0:
                                print('회피 완료')
                                avoid_start = False
                                
        except KeyboardInterrupt:
                print("프로그램 종료 중...")
        finally:
                node.destroy_node()
                rclpy.shutdown()
                

if __name__ == "__main__":
    main()
