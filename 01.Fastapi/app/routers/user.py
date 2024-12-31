from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List
from pathlib import Path

router = APIRouter()

# 기준 디렉터리 설정
BASE_DIR = Path("/home/khw/rep/final/02.Action_server/demo/output")


@router.get("/user/folders", response_model=List[str])
async def get_folders():
    """
    기준 디렉터리의 하위 폴더를 최신순으로 반환
    """
    if not BASE_DIR.exists():
        raise HTTPException(status_code=404, detail="Output 디렉터리를 찾을 수 없습니다.")

    folders = sorted(
        [folder.name for folder in BASE_DIR.iterdir() if folder.is_dir()],
        reverse=True
    )
    return folders


@router.get("/user/folders/{folder_name}/subfolders", response_model=List[str])
async def get_subfolders(folder_name: str):
    """
    선택한 폴더의 하위 폴더 (event, results)를 반환
    """
    folder_path = BASE_DIR / folder_name
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder를 찾을 수 없습니다.")

    return [subfolder.name for subfolder in folder_path.iterdir() if subfolder.is_dir()]


@router.get("/user/folders/{folder_name}/event-images", response_model=List[str])
async def get_event_images(folder_name: str):
    """
    event 폴더 내 이미지 파일 목록 반환
    """
    event_path = BASE_DIR / folder_name / "event"
    if not event_path.exists():
        raise HTTPException(status_code=404, detail="Event 폴더를 찾을 수 없습니다.")

    images = sorted([f"/images/{folder_name}/event/{img.name}" for img in event_path.glob("*.jpg")])
    if not images:
        raise HTTPException(status_code=404, detail="Event 폴더에 이미지가 없습니다.")
    return images


@router.get("/images/{folder_name}/event/{image_name}")
async def get_event_image(folder_name: str, image_name: str):
    """
    특정 Event 이미지 서빙
    """
    image_path = BASE_DIR / folder_name / "event" / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return FileResponse(image_path)


@router.get("/user/folders/{folder_name}/results-images", response_model=List[str])
async def get_results_images(folder_name: str):
    """
    results 폴더 내 이미지 파일 목록 반환
    """
    results_path = BASE_DIR / folder_name / "results"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results 폴더를 찾을 수 없습니다.")

    images = sorted([f"/images/{folder_name}/results/{img.name}" for img in results_path.glob("*.png")])
    if not images:
        raise HTTPException(status_code=404, detail="Results 폴더에 이미지가 없습니다.")
    return images


@router.get("/images/{folder_name}/results/{image_name}")
async def get_results_image(folder_name: str, image_name: str):
    """
    특정 Results 이미지 서빙
    """
    image_path = BASE_DIR / folder_name / "results" / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return FileResponse(image_path)
