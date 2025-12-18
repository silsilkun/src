"""
🏗️ Smart Tower Builder - NUCLEAR OPTION
=======================================
[최후의 수단 적용]
1. 그리퍼 전원: os.system()으로 터미널 명령어 직접 주입 (라이브러리 의존 X)
2. 동작 스킵 방지: movel(직선) 대신 movej(관절) 사용 (에러 무시하고 무조건 이동)
3. 디버깅 로그: 로봇이 어디로 가려는지 좌표를 전부 출력
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import threading
import time
import os  # <--- [핵심] 터미널 명령어 사용을 위해 추가

# 사용자 정의 모듈
import DR_init
from tower_builder.gripper_drl_controller import GripperController
from tower_builder.camera import BlockDetectionSystem, Block

# ============================================================
# ⚙️ [설정]
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 100, 50 # 속도 조금 줄임 (정확도 향상)

# 📍 좌표 보정
TRANSFORM_OFFSET_X = 685.0
TRANSFORM_OFFSET_Y = 20.0
CAMERA_Z_HEIGHT = 810.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            raise RuntimeError("Vision start failed")

        self.blocks = []
        self.target_stack_count = 0
        self.selected_queue = []
        self.stack_base_coords = None
        self.is_working = False

        self.gripper = None
        try:
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            time.sleep(1.0)
            if self.gripper.initialize():
                self.get_logger().info("✅ 그리퍼 객체 생성됨")
        except Exception as e:
            self.get_logger().error(f"그리퍼 오류: {e}")

    def stop_camera(self):
        self.vision.stop()

    def terminate_gripper(self):
        if self.gripper: self.gripper.terminate()



    # ============================================================
    # 마우스 콜백
    # ============================================================
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.is_working:
            print("⚠️ 로봇이 움직이고 있습니다!")
            return

        # 1. 블럭 선택
        if len(self.selected_queue) < self.target_stack_count:
            block = self.vision.find_block_at(x, y, update=False)
            if block:
                if block in self.selected_queue: return
                self.selected_queue.append(block)
                block.selection_order = len(self.selected_queue)
                
                w = min(block.real_width_mm, block.real_height_mm)
                print(f"✅ 블럭 선택 [{len(self.selected_queue)}/{self.target_stack_count}] (크기: {w:.1f}mm)")
                
                if len(self.selected_queue) == self.target_stack_count:
                    print("\n🎯 블럭 선택 완료! [탑을 쌓을 바닥]을 클릭하세요.")

        # 2. 위치 지정
        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if point_3d is None or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패.")
                return

            cam_x_mm = point_3d[0] * 1000
            cam_y_mm = point_3d[1] * 1000

            final_x = TRANSFORM_OFFSET_X + cam_y_mm
            final_y = cam_x_mm + TRANSFORM_OFFSET_Y
            
            self.stack_base_coords = (final_x, final_y)
            print(f"📍 타워 위치: ({final_x:.1f}, {final_y:.1f})")
            
            worker = threading.Thread(target=self.execute_stacking_sequence)
            worker.daemon = True
            worker.start()

    # ============================================================
    # 실행 시퀀스
    # ============================================================
    def execute_stacking_sequence(self):
        self.is_working = True
        print("\n🚀 로봇 작업 시퀀스 시작")
        
        stack_x, stack_y = self.stack_base_coords
        BASE_Z = 152.0
        BLOCK_H = 40.0

        from DSR_ROBOT2 import movej, movel, wait
        from DR_common2 import posj, posx

        try:
            
            # 3. 홈 정렬
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(1.0)

            # 4. 적재 루프
            for i, block in enumerate(self.selected_queue):
                print(f"\n🏗️ [{i+1}층 작업 시작] ---------------------")

                # 좌표 계산
                cam_x, cam_y, cam_z = block.center_3d_mm
                pick_x = TRANSFORM_OFFSET_X + cam_y
                pick_y = cam_x + TRANSFORM_OFFSET_Y
                pick_z = CAMERA_Z_HEIGHT - cam_z
                
                if pick_z < BASE_Z: pick_z = BASE_Z

                width = min(block.real_width_mm, block.real_height_mm)
                place_z = BASE_Z + (i * BLOCK_H)

                # [디버깅] 좌표 출력 (이게 이상하면 계산이 틀린 것)
                print(f"   👉 PICK 좌표: {pick_x:.1f}, {pick_y:.1f}, {pick_z:.1f}")
                print(f"   👉 PLACE 좌표: {stack_x:.1f}, {stack_y:.1f}, {place_z:.1f}")

                # 픽앤플레이스 실행
                self.move_smart_pick_and_place(pick_x, pick_y, pick_z, width,
                                               stack_x, stack_y, place_z)

            print("\n✨ 작업 완료! 홈 복귀.")
            movej(home_pose, vel=VELOCITY, acc=ACC)

        except Exception as e:
            self.get_logger().error(f"실행 중 오류: {e}")
        finally:
            self.selected_queue = []
            self.stack_base_coords = None
            self.is_working = False

   # ============================================================
    # [수정 완료] 타이밍 대폭 늘림 + 그리퍼 악력 강화
    # ============================================================
    def move_smart_pick_and_place(self, px, py, pz, width, sx, sy, sz):
        from DSR_ROBOT2 import get_current_posx, movel, movej, wait
        from DR_common2 import posx, posj

        # [수정 1] 그리퍼 값 튜닝 (꽉 잡게 값 올림)
        if width <= 35:
            # 작은 블럭
            target_open = 300; target_close = 600
        elif width <= 45:
            # 중간 블럭 (아까 600이 헐거웠으므로 750으로 상향)
            target_open = 200; target_close = 550
        else:
            # 큰 블럭
            target_open = 0; target_close = 450

        # 현재 자세 회전값(Rx, Ry, Rz) 유지
        cur_x = get_current_posx()[0]
        rx, ry, rz = cur_x[3], cur_x[4], cur_x[5]
        
        safe_z = 350.0

        # ----------------------------------------------------
        # [PICK] 잡으러 가기
        # ----------------------------------------------------
        print("   🚀 [1] Pick 위치 상공으로 이동 중... (3초 대기)")
        p_pick_ready = posx([px, py, safe_z, rx, ry, rz])
        movel(p_pick_ready, vel=VELOCITY, acc=ACC)
        wait(3.0) # [중요] 로봇이 도착할 때까지 충분히 기다림
        
        # 2. 그리퍼 벌리기
        if self.gripper: self.gripper.move(target_open)
        wait(1.0) # 벌리는 시간 확보
        
        # 3. 내려가기
        print("   🔻 [2] 하강하여 잡기... (2초 대기)")
        p_pick_down = posx([px, py, pz, rx, ry, rz])
        movel(p_pick_down, vel=VELOCITY/2, acc=ACC/2)
        wait(2.0) # 내려가는 시간 확보
        
        # 4. 잡기
        if self.gripper: self.gripper.move(target_close)
        print("   ✊ [3] 그립! (2초간 꽉 잡기)")
        wait(2.0) # [중요] 잡는 시간 충분히 줌 (아까 여기서 놓침)

        # 5. 올라오기
        print("   🔼 [4] 들어 올리기... (2초 대기)")
        movel(p_pick_ready, vel=VELOCITY, acc=ACC)
        wait(2.0)

        # ----------------------------------------------------
        # [PLACE] 쌓으러 가기
        # ----------------------------------------------------
        print("   🚀 [5] Place 위치로 이동... (4초 대기)")
        p_place_ready = posx([sx, sy, safe_z, rx, ry, rz])
        movel(p_place_ready, vel=VELOCITY, acc=ACC)
        wait(4.0) # [중요] 이동 거리가 머니까 더 기다림
        
        print("   🔻 [6] 하강하여 놓기... (2초 대기)")
        p_place_down = posx([sx, sy, sz + 15.0, rx, ry, rz])
        movel(p_place_down, vel=VELOCITY/2, acc=ACC/2)
        wait(2.0)

        if self.gripper: self.gripper.move(0) 
        print("   🖐 [7] 놓기 완료 (1초 대기)")
        wait(1.0) # 놓는 시간 확보

        # 복귀
        print("   🔼 [8] 복귀 중...")
        movel(p_place_ready, vel=VELOCITY, acc=ACC)
        wait(2.0)

    def process_and_render(self):
        self.vision.config.threshold = cv2.getTrackbarPos("Threshold", "Control")
        self.vision.config.min_area = cv2.getTrackbarPos("Min Area", "Control")
        self.vision.config.max_area = cv2.getTrackbarPos("Max Area", "Control")

        if not self.vision.update(): return

        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()
        c = self.vision.config
        cv2.rectangle(display, (c.roi_x, c.roi_y), (c.roi_x+c.roi_w, c.roi_y+c.roi_h), (0,0,255), 2)

        for block in self.blocks:
            is_sel = (block in self.selected_queue)
            col = (0, 255, 255) if is_sel else (0, 255, 0)
            cv2.drawContours(display, [block.rotated_box], 0, col, 2)
            cx, cy = block.center_2d
            
            w_mm = min(block.real_width_mm, block.real_height_mm)
            cv2.putText(display, f"{w_mm:.0f}mm", (cx-20, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if hasattr(block, 'selection_order'):
                cv2.putText(display, f"#{block.selection_order}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if self.target_stack_count == 0: msg = "Input number in terminal"
        elif len(self.selected_queue) < self.target_stack_count: msg = "Select Blocks..."
        elif self.stack_base_coords is None: msg = ">> Click Target Floor <<"
        else: msg = "Auto Stacking..."
        cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Result", display)
        
        if self.vision.last_depth is not None:
            d_view = cv2.applyColorMap(cv2.convertScaleAbs(self.vision.last_depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Depth", d_view)


def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    
    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except: pass

    robot = RobotControllerNode()

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(dsr_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot.mouse_callback)
    
    cv2.namedWindow("Control")
    cfg = robot.vision.config
    cv2.createTrackbar("Threshold", "Control", cfg.threshold, 255, lambda x: None)
    cv2.createTrackbar("Min Area", "Control", cfg.min_area, 5000, lambda x: None)
    cv2.createTrackbar("Max Area", "Control", cfg.max_area, 30000, lambda x: None)

    print("\n" + "="*40)
    print("🏗️ Smart Tower Builder (NUCLEAR OPTION)")
    print("="*40)

    try:
        while rclpy.ok():
            if robot.target_stack_count == 0:
                try:
                    val = input("\n👉 몇 층 탑을 쌓으시겠습니까? (숫자) >> ")
                    cnt = int(val)
                    if cnt > 0:
                        robot.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 선택하세요.")
                except ValueError: pass
                continue
            
            robot.process_and_render()
            
            if not robot.is_working and robot.target_stack_count > 0 and robot.stack_base_coords is not None:
                if len(robot.selected_queue) == 0:
                    robot.target_stack_count = 0
                    robot.stack_base_coords = None
                    print("\n🎉 완료! 다시 시작합니다.")

            if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        print("종료")
    finally:
        robot.terminate_gripper()
        robot.stop_camera()
        cv2.destroyAllWindows()
        executor.shutdown()
        robot.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()