// 요소 가져오기
const output = document.getElementById("output");
const toggleButton = document.getElementById("toggle-admin");

// WebSocket 연결
const ws = new WebSocket(`ws://${window.location.host}/ws/keyboard`);

// WebSocket 메시지 수신
ws.onmessage = (event) => {
    output.textContent = event.data; // 서버로부터 수신된 메시지 출력
};

// 키보드 이벤트 감지 및 WebSocket으로 전송
document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase(); // 입력된 키를 소문자로 변환
    if (["w", "a", "s", "d"].includes(key)) {
        ws.send(key); // 키 입력을 WebSocket으로 서버에 전송
    }
});

// 관리자 권한 On/Off 버튼 클릭 이벤트
toggleButton.addEventListener("click", () => {
    fetch("/toggle-admin") // 서버에 권한 토글 요청
        .then((response) => response.json())
        .then((data) => {
            // 버튼 상태 업데이트
            toggleButton.textContent = `관리자 권한 ${data.status}`;
            toggleButton.style.backgroundColor = data.status === "On" ? "red" : "blue";

            // 상태 출력 업데이트
            output.textContent = `관리자 권한이 ${data.status} 상태로 변경되었습니다.`;
        })
        .catch((error) => {
            console.error("관리자 권한 토글 오류:", error);
            output.textContent = "오류가 발생했습니다. 다시 시도해 주세요.";
        });
});
