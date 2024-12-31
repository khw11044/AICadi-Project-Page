from fastapi import APIRouter, WebSocket
from fastapi.responses import FileResponse
import asyncio
import socket

router = APIRouter()

# 소켓 서버 설정
HOST = "0.0.0.0"
PORT = 5000

@router.websocket("/ws/robot")
async def websocket_robot(websocket: WebSocket):
    """
    WebSocket을 통해 소켓 명령어를 실시간으로 클라이언트에 전송
    """
    await websocket.accept()
    print("WebSocket 연결 시작")

    # UDP 소켓 서버 초기화
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind((HOST, PORT))
    print(f"UDP 소켓 서버 시작: {HOST}:{PORT}")

    try:
        while True:
            udp_socket.settimeout(0.1)
            try:
                # UDP 데이터 수신
                data, _ = udp_socket.recvfrom(1024)
                command = data.decode()
                
                if command in ['nothing', 'ready', 'stop', 'emergency', 'no person']:
                    # 수신된 명령을 클라이언트에 전송
                    print(f"수신된 명령어: {command}")
                    await websocket.send_text(command)
            except socket.timeout:
                await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket 오류: {e}")
    finally:
        udp_socket.close()
        await websocket.close()
        print("WebSocket 연결 종료")


@router.get("/robot")
async def robot_page():
    """로봇 모드 페이지 서빙"""
    return FileResponse("static/robot/robot.html")
