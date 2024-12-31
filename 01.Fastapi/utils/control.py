

def human_follower(decoded_data):
    x, y, _, _, _, _, _, _, _, _, way, dist = map(float, decoded_data.split(","))
    
    # speed = 0.9 if abs(dist) > 0.9 else round(abs(dist)+0.02, 2)
    speed = 0.05
    way = 0.4 if abs(way) > 0.4 else round(abs(way)+0.02, 2)
    
    comm = ''
    
    # 방향 제어
    if x == 1:  # 오른쪽 회전
        angular_z = -way
        comm += f'우회전: {angular_z} / '
    elif x == -1:  # 왼쪽 회전
        angular_z = way
        comm += f'좌회전: {angular_z} / '
    else:  # 중립
        angular_z = 0.0
        comm += f'그대로: {angular_z} / '
        
    if y == 1:  # 앞으로
        linear_x = speed
        comm += f'전진: {linear_x}'
    elif y == -1:  # 뒤로
        linear_x = -speed
        comm += f'후진: {linear_x}'
    else:  # 중립
        linear_x = 0.0
        comm += f'정지: {linear_x}'
    
    print('--------------------')
    print(comm)
    print('--------------------')
    return linear_x, angular_z