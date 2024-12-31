import rclpy
import socket
from AvoidtotheRightFunctions import ObstacleAvoidanceNode

# 서버 설정
SERVER_IP = "192.168.0.126"     # FastAPI 서버 IP
SERVER_PORT = 5000              # FastAPI 서버와 통신할 포트

def main():
    rclpy.init()
    buffer_distance = 0.25
    right_cnt = 0
    left_cnt = 0
    # 노드 생성
    node = ObstacleAvoidanceNode()
    
    # Socket 설정 (UDP 통신)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Socket 통신 시작: {SERVER_IP}:{SERVER_PORT}")
    
    avoid_start = False
    start_half = False

    try:
        # 라이다 데이터 수신 대기
        while not node.ranges:
            rclpy.spin_once(node, timeout_sec=0.1)
        print("LaserScan data received. 시스템 시작.")

        
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # 3. 장애물과 일정 거리 유지하면서 
            if node.range_front_left >= node.obstacle_distance_threshold + buffer_distance:
                # 3.1 오른쪽으로 회피한 기록이 있으면 오른쪽 간 만큼 왼쪽 가기 
                if node.angularZ_count > 0:
                    node.AngleRestoration()
                    left_cnt += 1
                # 3.2 오른쪽 회피한 만큼 왼쪽 복구를 다했다면 왼쪽 복구 만큼 왼쪽 더 가기 
                elif avoid_start and right_cnt > 0 and left_cnt > 0:
                    node.AvoidToTheLeft()
                    right_cnt -= 1
                    
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
                        # 장애물 감지 시 Socket으로 메시지 전송
                        print("장애물 감지: 전방에 장애물 인식")
                        sock.sendto("전방에 장애물 인식".encode(), (SERVER_IP, SERVER_PORT))

                
                # 2.2 장애물 오른쪽 회피 
                node.AvoidToTheRight()
                right_cnt += 1
            
            if avoid_start and node.angularZ_count == 0:
                start_half = True 
            
            if avoid_start and node.angularZ_count == 0 and right_cnt == 0 and left_cnt == 0:
                # 장애물 회피 완료 메시지 전송
                print("장애물 회피 완료")
                sock.sendto("장애물 회피 완료".encode(), (SERVER_IP, SERVER_PORT))

                avoid_start = False
                start_half = False
            
                
    except KeyboardInterrupt:
        print("프로그램 종료 중...")

    finally:
        # 정리 작업
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        sock.close()
        print("시스템 종료.")

if __name__ == "__main__":
    main()
