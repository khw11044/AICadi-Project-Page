import time 

send_interval = 0.05 
last_sent_time = 0
last_data = None



# 로봇 제어 함수
def control_robot(sock, ROBOT_IP, ROBOT_PORT, cur_action):
    global send_interval, last_sent_time, last_data
    prev_action = cur_action  # 이전 x 값 저장


    # 변경된 값이 있거나 송신 주기가 지난 경우에만 데이터 전송
    if cur_action != prev_action or time.time() - last_sent_time >= send_interval:
        
        data = f"{cur_action}"

        # 동일 데이터 중복 전송 방지
        if data != last_data:
            sock.sendto(data.encode(), (ROBOT_IP, ROBOT_PORT))
            last_data = data
            last_sent_time = time.time()

            # 디버깅 출력
            print("전송 데이터:", data)

