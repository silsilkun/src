#리스트
import v9
import threading
import time

# 1️⃣ v8 실행
v9.start_system()

# 2️⃣ GUI 실행
threading.Thread(target=v9.run_gui, daemon=True).start()

print("GUI 실행됨, 클릭하세요")
input("엔터 누르면 결과 출력")

# 3️⃣ 값 가져오기
blocks = v9.get_clicked_blocks()
floors = v9.get_clicked_floor_points()

print("블록 개수:", blocks)
print("바닥 개수:", floors)

# 4️⃣ 종료
v9.stop_system()


