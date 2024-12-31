import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import Twist
from AvoidtotheRightFunctions import ObstacleAvoidanceNode
# args=None
def main():
        rclpy.init()
        buffer_distance = 0.25
        right_cnt = 0
        left_cnt = 0
        #노드 생성
        node = ObstacleAvoidanceNode()
                # 라이다 데이터가 수신될 때까지 대기
        while not node.ranges:  # ranges가 빈 상태인지 확인
                rclpy.spin_once(node, timeout_sec=0.1)

        print("LaserScan data received.")
        avoid_start = False
        start_half = False
        # 장애물 거리 출력
        try:
                while rclpy.ok():
                        rclpy.spin_once(node, timeout_sec=0.1)  # 콜백 처리
                        
                        # 3. 장애물과 일정 거리 유지하면서 
                        if node.range_front_left >= node.obstacle_distance_threshold + buffer_distance:
                                # 3.1 오른쪽으로 회피한 기록이 있으면 오른쪽 간 만큼 왼쪽 가기 
                                if node.angularZ_count > 0:
                                        # print('복구 중 / angularZ_count:', node.angularZ_count)
                                        node.AngleRestoration()
                                        left_cnt += 1
                                        
                                # 3.2 오른쪽 회피한 만큼 왼쪽 복구를 다했다면 왼쪽 복구 만큼 왼쪽 더 가기 
                                elif avoid_start and right_cnt > 0 and left_cnt > 0:
                                        node.AvoidToTheLeft()
                                        right_cnt -= 1
                                        # print(f"복구 왼쪽 회전 횟수: {right_cnt}")
                                
                                # 3.3 오른쪽 회피 후 왼쪽 복구 하고 그 이후 오른쪽 복구 
                                elif avoid_start and right_cnt <= 10 and left_cnt != 0:
                                        while left_cnt != 0:
                                                node.AvoidToTheRight2()
                                                left_cnt -= 1
                                                print(f"복구 오른쪽 회전 횟수: {left_cnt}")
                                        
                        # 2. 장애물이 앞에 있을 때 
                        if node.range_front_left < node.obstacle_distance_threshold and not start_half:
                                # 2.1 첫 장애물 발견 
                                if not avoid_start:
                                        avoid_start = True
                                        right_cnt = 0
                                        print('우측 회피를 시작합니다.')
                                        
                                # 2.2 장애물 오른쪽 회피 
                                node.AvoidToTheRight()
                                right_cnt += 1
                                # print('우측 회피 중 / angularZ_count:', node.angularZ_count)
                                # print('right_cnt ->', right_cnt)
                        
                        # 1. 앞으로 쭉 가다가 
                        else:
                                print('앞으로 가기.')
                                node.move_forward()
                                
                        if avoid_start and node.angularZ_count == 0:
                                start_half = True 
                                
                        print('node.angularZ_count, right_cnt, left_cnt')
                        print(node.angularZ_count, right_cnt, left_cnt)
                             
                        if avoid_start and node.angularZ_count == 0 and right_cnt == 0 and left_cnt == 0:
                                print('회피 완료')
                                avoid_start = False
                                start_half = False
                                
        except KeyboardInterrupt:
                print("프로그램 종료 중...")
        finally:
                node.destroy_node()
                rclpy.shutdown()
                

if __name__ == "__main__":
    main()
