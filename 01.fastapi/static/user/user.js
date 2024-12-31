document.addEventListener("DOMContentLoaded", () => {
    const folderContainer = document.getElementById("folder-container");
    const displayContainer = document.getElementById("display-container");
    let currentImageIndex = 0;
    let imageList = [];
    let slideshowInterval;

    // 1. 상위 폴더 목록 불러오기
    fetch("/user/folders")
        .then(response => response.json())
        .then(folders => {
            folders.forEach(folder => {
                const listItem = document.createElement("li");
                listItem.textContent = folder;
                listItem.style.cursor = "pointer";

                // 하위 폴더를 담을 리스트
                const subList = document.createElement("ul");
                subList.style.display = "none";
                subList.style.marginLeft = "20px";

                // 상위 폴더 클릭 이벤트
                listItem.addEventListener("click", () => {
                    if (subList.childNodes.length === 0) {
                        fetch(`/user/folders/${folder}/subfolders`)
                            .then(response => response.json())
                            .then(subfolders => {
                                subList.innerHTML = "";
                                subfolders.forEach(sub => {
                                    const subItem = document.createElement("li");
                                    subItem.textContent = sub;
                                    subItem.style.cursor = "pointer";

                                    // Event 또는 Results 클릭 이벤트
                                    subItem.onclick = () => {
                                        if (sub === "event") loadEventImages(folder);
                                        if (sub === "results") loadResultsSlideshow(folder);
                                    };
                                    subList.appendChild(subItem);
                                });
                                subList.style.display = "block";
                            });
                    } else {
                        subList.style.display = subList.style.display === "none" ? "block" : "none";
                    }
                });

                folderContainer.appendChild(listItem);
                folderContainer.appendChild(subList);
            });
        })
        .catch(error => {
            console.error("폴더 목록 불러오기 실패:", error);
            folderContainer.innerHTML = "<li>폴더를 불러오지 못했습니다.</li>";
        });

    // 2. Event 폴더 이미지 슬라이드 불러오기
    function loadEventImages(folder) {
        clearSlideshow(); // 기존 슬라이드쇼 종료
        fetch(`/user/folders/${folder}/event-images`)
            .then(response => response.json())
            .then(images => {
                if (images.length === 0) {
                    displayContainer.innerHTML = "<p>이미지가 없습니다.</p>";
                    return;
                }

                imageList = images;
                currentImageIndex = 0;
                showImage();

                // 키보드 이벤트 추가
                document.removeEventListener("keydown", handleImageSlide);
                document.addEventListener("keydown", handleImageSlide);
            })
            .catch(error => {
                console.error("이미지 불러오기 실패:", error);
                displayContainer.innerHTML = "<p>이미지를 불러오지 못했습니다.</p>";
            });

        function showImage() {
            displayContainer.innerHTML = "";
            const img = document.createElement("img");
            img.src = imageList[currentImageIndex];
            img.style.maxWidth = "100%";
            img.style.borderRadius = "10px";
            displayContainer.appendChild(img);

            // 화살표 추가
            addArrows();
        }

        function addArrows() {
            const leftArrow = createArrow("left", () => {
                currentImageIndex = (currentImageIndex - 1 + imageList.length) % imageList.length;
                showImage();
            });
            const rightArrow = createArrow("right", () => {
                currentImageIndex = (currentImageIndex + 1) % imageList.length;
                showImage();
            });

            displayContainer.appendChild(leftArrow);
            displayContainer.appendChild(rightArrow);
        }

        function createArrow(direction, onClick) {
            const arrow = document.createElement("button");
            arrow.textContent = direction === "left" ? "◀" : "▶";
            arrow.className = "arrow";
            arrow.style.position = "absolute";
            arrow.style.top = "50%";
            arrow.style[direction] = "20px";
            arrow.style.transform = "translateY(-50%)";
            arrow.style.zIndex = "10";
            arrow.onclick = onClick;
            return arrow;
        }

        function handleImageSlide(e) {
            if (e.key === "ArrowRight") {
                currentImageIndex = (currentImageIndex + 1) % imageList.length;
                showImage();
            } else if (e.key === "ArrowLeft") {
                currentImageIndex = (currentImageIndex - 1 + imageList.length) % imageList.length;
                showImage();
            }
        }
    }

    // 3. Results 폴더 이미지 자동 슬라이드쇼
    function loadResultsSlideshow(folder) {
        clearSlideshow(); // 기존 슬라이드쇼 종료
        fetch(`/user/folders/${folder}/results-images`)
            .then(response => response.json())
            .then(images => {
                if (images.length === 0) {
                    displayContainer.innerHTML = "<p>이미지가 없습니다.</p>";
                    return;
                }

                imageList = images;
                currentImageIndex = 0;
                startSlideshow();
            })
            .catch(error => {
                console.error("Results 이미지 불러오기 실패:", error);
                displayContainer.innerHTML = "<p>이미지를 불러오지 못했습니다.</p>";
            });

        function startSlideshow() {
            displayContainer.innerHTML = "";
            const img = document.createElement("img");
            img.style.maxWidth = "100%";
            img.style.borderRadius = "10px";
            displayContainer.appendChild(img);

            slideshowInterval = setInterval(() => {
                img.src = imageList[currentImageIndex];
                currentImageIndex = (currentImageIndex + 1) % imageList.length;
            }, 50); // 0.5초마다 이미지 변경
        }
    }

    // 4. 기존 슬라이드쇼 종료 함수
    function clearSlideshow() {
        clearInterval(slideshowInterval);
        slideshowInterval = null;
        imageList = [];
        currentImageIndex = 0;
        displayContainer.innerHTML = "";
    }
});
