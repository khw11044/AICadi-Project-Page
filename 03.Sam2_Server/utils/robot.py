import time 

send_interval = 0.05 
last_sent_time = 0
last_data = None

# 초기화
x, y = 127, 127
blue, green, red, yellow, left_top, right_top = 0, 0, 0, 0, 0, 0
select_key, start_key = 0, 0
way2, dist2 = 0,0

# 로봇 제어 함수
def control_robot(way, distances, sock, ROBOT_IP, ROBOT_PORT):
    global x, y, way2, dist2 
    global blue, green, red, yellow, left_top, right_top, select_key, start_key
    global last_data, last_sent_time
    comm = '그대로'

    init_ph, cur_ph, frame_h = distances
    dist = init_ph - cur_ph
    
    prev_way = way2  # 이전 x 값 저장
    prev_dist = dist2  # 이전 x 값 저장
    if way < -90:
        comm = '좌회전'
        x = -1
    elif way > 90:
        comm = '우회전'
        x = 1
    else:
        comm = '그대로'
        x = 127
    
    if init_ph - frame_h <= dist < -10:
        y = -1  # 뒤로 가 
        comm += '/후진'
    elif -10 <= dist < 50:
        y = 0.0 
        comm += '/정지'
    else:
        y = 1   # 앞으로 가 
        comm += '/전진'
    

    way2 = round((way / 100)**2 / 10, 1)
    dist2 = round(dist*3 / 1000, 1)
        
    # 변경된 값이 있거나 송신 주기가 지난 경우에만 데이터 전송
    if way2 != prev_way or dist2 != prev_dist:
        print(comm)
        print('방향수치:', way2)
        print('거리수치:', dist2)
        data = f"{x},{y},{blue},{green},{red},{yellow},{left_top},{right_top},{select_key},{start_key},{way2},{dist2}"

        # 동일 데이터 중복 전송 방지
        if data != last_data:
            sock.sendto(data.encode(), (ROBOT_IP, ROBOT_PORT))
            last_data = data
            last_sent_time = time.time()

            # # 디버깅 출력
            # print("전송 데이터:", data)