#딕셔너리
import v9
import threading
import time

# 1️⃣ 시스템 시작 (카메라 + 데이터 구조 생성)
v9.start_system()

# 2️⃣ GUI 실행 (별도 스레드)
threading.Thread(target=v9.run_gui, daemon=True).start()

print("GUI 실행됨. 블록/바닥 클릭하세요")

# 3️⃣ 사용자가 클릭할 시간 확보
input("엔터 누르면 결과 출력")


# 4️⃣ 값 가져오기 (이제 에러 안 남)
blocks = v9.get_block_summaries()
floors = v9.get_floor_summaries()

print("블록 요약:")
for b in blocks:
    print(b)

print("바닥 요약:")
for f in floors:
    print(f)

print(blocks, floors)

# 5️⃣ 시스템 종료
v9.stop_system()

# def vtestData():
#     v9.start_system()

#     # 2️⃣ GUI 실행 (별도 스레드)
#     threading.Thread(target=v9.run_gui, daemon=True).start()

#     print("GUI 실행됨. 블록/바닥 클릭하세요")

#     # 3️⃣ 사용자가 클릭할 시간 확보
#     input("엔터 누르면 결과 출력")


#     # 4️⃣ 값 가져오기 (이제 에러 안 남)
#     blocks = v9.get_block_summaries()
#     floors = v9.get_floor_summaries()

#     print("블록 요약:")
#     for b in blocks:
#         print(b)

#     print("바닥 요약:")
#     for f in floors:
#         print(f)

#     return blocks, floors

#     # 5️⃣ 시스템 종료
#     v9.stop_system()

#   조작방법:
#    - 블록 클릭: 상세 정보 + 리스트 저장
#    - 빈 공간 클릭: 깊이 확인
#    - GUI 실행됨. 블록/바닥 클릭하세요
#    - 'p' 키: 저장된 블록 요약
#    - 'c' 키: 저장 리스트 초기화
#    - ESC: 종료
#    딕셔너리 형태로 좌표 각도 변을 받아올 수 있다
#import v9
#v9.start_system()
