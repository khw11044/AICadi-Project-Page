from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import sys



# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# 라우터 가져오기
from app.routers import admin, user, robot

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(admin.router)
app.include_router(user.router)
app.include_router(robot.router)

# 홈 화면 경로
@app.get("/")
async def home():
    return FileResponse("static/index/index.html")

# 사용자 모드 페이지
@app.get("/user")
async def user_page():
    return FileResponse("static/user/user.html")

# 관리자 모드 페이지
@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin/admin.html")

# 직접 실행 시 uvicorn 서버 실행
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
