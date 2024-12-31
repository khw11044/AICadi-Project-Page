import socket
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import asyncio
import signal
import atexit
from utils.control import human_follower  # 명령어 해석 함수

# 설정 상수
HOST = "0.0.0.0"
PORT = 5000
MAX_IDLE_TIME = 60 * 5  # 최대 유휴 시간 (초)
READY_DURATION = 30  # 준비 상태 지속 시간
STOP_DURATION = 5    # 멈춤 지속 시간


class SocketControl:
    def __init__(self):
        """ROS2 노드 초기화 및 설정"""
        if not rclpy.utilities.ok():
            rclpy.init()
        self.node = rclpy.create_node('socket_control_node')
        self.publisher = self.node.create_publisher(Twist, 'cmd_vel', 10)
        self.twist = Twist()

        # 상태 변수 초기화
        self.state_manager = {
            "linear_x": 0.0, "angular_z": 0.0,
            "last_received_time": time.time(),
            "ready": False, "ready_time": None,
            "stop": False, "stop_time": None,
            "avoid": False,  # 장애물 회피 상태
            "state": True
        }
        print("ROS2 노드 초기화 완료.")

    def cleanup(self):
        """프로그램 종료 시 리소스 정리"""
        self.publisher.publish(Twist())  # 로봇 정지
        print("ROS2: 정지 메시지 발행")
        rclpy.shutdown()
        print("ROS2: 노드 종료")
        print("프로그램 종료 및 리소스 정리 완료.")

    async def start_socket_server(self):
        """UDP 소켓을 통해 데이터를 수신하고 로봇을 제어"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((HOST, PORT))
        print(f"소켓 서버 시작: {HOST}:{PORT}")

        try:
            while rclpy.ok():
                sock.settimeout(0.1)
                try:
                    data, _ = sock.recvfrom(1024)  # UDP 데이터 수신
                    command = data.decode()
                    print("수신 명령어:", command)

                    self.state_manager["last_received_time"] = time.time()

                    # 장애물 감지 및 회피 처리
                    if command == "전방에 장애물 인식":
                        print("장애물 감지: 회피 모드로 전환합니다.")
                        self.state_manager["avoid"] = True
                    elif command == "장애물 회피 완료":
                        print("장애물 회피 완료: 회피 모드를 종료합니다.")
                        self.state_manager["avoid"] = False
                    else:
                        self.process_command(command)

                except socket.timeout:
                    pass

                # 유휴 시간 확인
                if time.time() - self.state_manager["last_received_time"] > MAX_IDLE_TIME:
                    self.node.get_logger().info("2분 이상 데이터 수신 없음. 프로그램 종료.")
                    break

                # Twist 메시지 업데이트 및 발행
                self.update_twist()
                self.publisher.publish(self.twist)

                # ROS 노드 스핀 (비동기)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("종료 요청을 받았습니다.")
        finally:
            self.cleanup()
            sock.close()

    def process_command(self, command):
        """수신된 명령을 처리하는 함수"""
        if self.state_manager["avoid"]:
            return  # 장애물 회피 중에는 다른 명령어를 처리하지 않음

        if command == "ready" and not self.state_manager["ready"]:
            if not self.state_manager["stop"]:
                print("스윙 준비 중입니다.... 스윙 촬영을 하겠습니다.")
                self.state_manager.update({
                    "ready": True, "ready_time": time.time(), "state": False
                })
        elif command == "no person":
            print("사람이 없어요.")
            self.state_manager.update({"ready": False, "linear_x": 0.0, "angular_z": 0.0})
        elif command == "stop":
            if not self.state_manager["stop"] or time.time() - self.state_manager["stop_time"] > STOP_DURATION:
                self.state_manager["stop"] = not self.state_manager["stop"]
                self.state_manager["stop_time"] = time.time()
                print("멈춤 상태" if self.state_manager["stop"] else "다시 시작")
        elif command == "nothing":
            if not self.state_manager["stop"]:
                self.state_manager["state"] = True
        else:
            # 움직임 명령어 처리
            if self.state_manager["state"]:
                self.state_manager["linear_x"], self.state_manager["angular_z"] = human_follower(command)
                
                # if self.state_manager["stop"]:
                #     self.state_manager["linear_x"] = 0.0

    def update_twist(self):
        """현재 상태에 따라 Twist 메시지 업데이트"""
        if self.state_manager["avoid"]:
            # # 장애물 회피 중: 로봇을 멈춤
            # self.twist.linear.x = 0.0
            # self.twist.angular.z = 0.0
            return
        elif self.state_manager["stop"]:
            # 멈춤 상태
            self.twist.linear.x = 0.0
            self.twist.angular.z = self.state_manager["angular_z"]
        else:
            # 일반 주행 상태
            self.twist.linear.x = self.state_manager["linear_x"]
            self.twist.angular.z = self.state_manager["angular_z"]


# 전역 인스턴스
socket_control = SocketControl()

# 종료 시 정리 함수 등록
atexit.register(socket_control.cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: socket_control.cleanup())
signal.signal(signal.SIGTERM, lambda sig, frame: socket_control.cleanup())
