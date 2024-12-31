const ws = new WebSocket("ws://" + window.location.host + "/ws/robot");

// 이미지 경로 매핑
const imagePaths = {
    "avoid": "/static/robot/face/avoid.png",
    "ready": "/static/robot/face/ready.png",
    "stop": "/static/robot/face/stop.png",
    "emergency": "/static/robot/face/emergency.png",
    "nothing": "/static/robot/face/following.png",
    "no person": "/static/robot/face/noperson.png"
};

// WebSocket 메시지 수신
ws.onmessage = (event) => {
    const command = event.data;
    console.log("명령어 수신:", command);

    const imageElement = document.getElementById("robot-image");

    // 명령어에 맞는 이미지 변경
    if (imagePaths[command]) {
        imageElement.src = imagePaths[command];
    } else {
        console.warn("알 수 없는 명령어:", command);
    }
};

ws.onopen = () => {
    console.log("WebSocket 연결됨");
};

ws.onclose = () => {
    console.log("WebSocket 연결 종료됨");
};
