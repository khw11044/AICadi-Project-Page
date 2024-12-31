from fastapi import APIRouter, WebSocket
from app.services.ros_control import init_ros_control
from app.services.socket_control import socket_control
import asyncio

router = APIRouter()

# 상태 변수: 관리자 권한 On/Off
admin_control = {"enabled": False}

# ROS2 제어 노드 초기화
ros_node = init_ros_control()

# 소켓 제어 작업을 위한 태스크 변수
socket_task = None


@router.websocket("/ws/keyboard")
async def websocket_keyboard(websocket: WebSocket):
    """
    WebSocket 엔드포인트: 관리자 권한 On일 때 키보드 입력 처리
    """
    await websocket.accept()
    print("WebSocket 연결 시작")
    try:
        while True:
            data = await websocket.receive_text()
            if admin_control["enabled"]:  # 관리자 권한 On 상태
                print(f"WebSocket 수신: {data}")
                ros_node.control_robot(data)  # ROS2 로봇 제어
                await websocket.send_text(f"키 입력: {data}")
            else:
                await websocket.send_text("관리자 권한 Off 상태입니다.")
    except Exception as e:
        print(f"WebSocket 오류: {e}")
    finally:
        print("WebSocket 연결 종료")


@router.get("/toggle-admin")
async def toggle_admin():
    """
    관리자 권한 On/Off 토글 엔드포인트
    """
    global socket_task

    # 관리자 권한 상태 변경
    admin_control["enabled"] = not admin_control["enabled"]
    status = "On" if admin_control["enabled"] else "Off"

    # 관리자 권한 Off 상태일 때 소켓 서버 실행
    if not admin_control["enabled"]:
        if socket_task is None or socket_task.done():
            print("관리자 권한 Off: 소켓 서버 시작")
            socket_task = asyncio.create_task(socket_control.start_socket_server())
    else:
        # 관리자 권한 On 상태일 때 소켓 서버 중지
        if socket_task and not socket_task.done():
            socket_task.cancel()
            print("관리자 권한 On: 소켓 서버 중지")

    return {"status": status}
