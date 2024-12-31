import rclpy
import socket
from AvoidtotheRightFunctions import ObstacleAvoidanceNode

# 서버 설정
SERVER_IP = "192.168.0.125"     # FastAPI 서버 IP
SERVER_PORT = 5000              # FastAPI 서버와 통신할 포트

def main():
    rclpy.init()

    # 노드 생성
    node = ObstacleAvoidanceNode()
    
    # Socket 설정 (UDP 통신)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Socket 통신 시작: {SERVER_IP}:{SERVER_PORT}")

    try:
        # 라이다 데이터 수신 대기
        while not node.ranges:
            rclpy.spin_once(node, timeout_sec=0.1)
        print("LaserScan data received. 시스템 시작.")

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # 장애물 거리 확인
            if node.range_front_left < node.obstacle_distance_threshold:
                # 장애물 감지 시 Socket으로 메시지 전송
                print("장애물 감지: 전방에 장애물 인식")
                sock.sendto("전방에 장애물 인식".encode(), (SERVER_IP, SERVER_PORT))

                # 장애물 회피 실행
                while node.range_front_left < node.obstacle_distance_threshold:
                    node.AvoidToTheRight()
                    rclpy.spin_once(node, timeout_sec=0.1)

                # 장애물 회피 완료 메시지 전송
                print("장애물 회피 완료")
                sock.sendto("장애물 회피 완료".encode(), (SERVER_IP, SERVER_PORT))

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
